import json
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass
from time import sleep, time
from typing import List, Set, Callable, Tuple

import chess.engine
import numpy as np
from reconchess import Square
from tqdm import tqdm

from Fianchetto.utilities import SEARCH_SPOTS, stockfish, simulate_move, simulate_sense, \
    generate_rbc_moves, generate_moves_without_opponent_pieces, force_promotion_to_queen
from Fianchetto.utilities.rbc_move_score import calculate_score, ScoreConfig, softmax
from Fianchetto.utilities.player_logging import create_sub_logger
from reconchess.utilities import move_actions

SCORE_ROUNDOFF = 1e-5
move_ind = 0
store = []
Qs = []
msi = []
esr = []
msr = []

@dataclass
class MoveConfig:
    mean_score_factor: float = 0.45  # relative contribution of a move's average outcome on its compound score
    min_score_factor: float = 0.45  # relative contribution of a move's worst outcome on its compound score
    max_score_factor: float = 0.1  # relative contribution of a move's best outcome on its compound score
    threshold_score_factor: float = 0.1  # fraction below best compound score in which any move will be considered
    sense_by_move: bool = False  # Use bonus score to encourage board set reduction by attempted moves
    force_promotion_queen: bool = True  # change all pawn-promotion moves to choose queen, otherwise it's often a knight


@dataclass
class TimeConfig:
    turns_to_plan_for: int = 7  # fixed number of turns over which the remaining time will be divided
    min_time_for_turn: float = 1.0  # minimum time to allocate for a turn
    time_for_sense: float = 0.8  # fraction of turn spent in choose_sense
    time_for_move: float = 0.2  # fraction of turn spent in choose_move
    calc_time_per_move: float = 0.001  # starting time estimate for move score calculation


# Add a hash method for chess.Board objects so that they can be tested for uniqueness. For our purposes, a unique EPD
#  string is adequate; it contains piece positions, castling rights, and en-passant square, but not turn counters.
chess.Board.__hash__ = lambda self: hash(self.fen())


# Create a cache key for the requested board and move (keyed based on the move that would result from that request)
def make_cache_key(board, color=None):
    if color and type(board) == str:
        board = chess.Board(board)
    elif type(board) == str:
        return board

    if color:
        board.turn = color
    return board.fen()


# Each "worker" runs a StockFish engine and waits for requested scores
def worker(request_queue, response_queue, score_config, num_threads):
    logger = create_sub_logger('lc0_queue_worker')
    b = stockfish.create_engine()
    while True:
        if not request_queue.empty():
            (board, moves) = request_queue.get()

            # exit()
            try:
                score = calculate_score(board_fen=board, b=b, moves=moves, score_config=score_config)
                # print('Q:',score[0])
                Qs.append(score[0])
                # print('type', type(score))
                # print(score)
            except Exception as e:
                print('Got in Exception:', e)
                logger.error('lc0 engine died while analysing (%s).',
                             board)
                # If the analysis crashes the engine, something went really wrong. This tends to happen when the
                #  board is not valid, but that is meant to be filtered in calculate_score. Just in case, do not
                #  re-query the engine, instead assign the move a conservative score (here: as though into check).
                response_queue.put({make_cache_key(board): score_config.into_check_score})
                b = stockfish.create_engine()
            else:
                response_queue.put({make_cache_key(board): score})
        else:
            sleep(0.001)


def create_strategy(
        move_config: MoveConfig = MoveConfig(),
        score_config: ScoreConfig = ScoreConfig(),
        time_config: TimeConfig = TimeConfig(),

        board_weight_90th_percentile: float = 0.99,
        boards_per_centipawn: int = 600,
        num_workers: int = 1,
        num_threads: int = None,

        checkmate_sense_override: bool = True,
        while_we_wait_extension: bool = True,
        load_cache_data: bool = False,
        rc_disable_pbar: bool = False
)\
        -> Tuple[Callable[[Set[str], bool, List[Square], List[chess.Move], float], Square],
                 Callable[[Set[str], bool, List[chess.Move], float], chess.Move],
                 Callable[[Set[str], bool], None],
                 Callable[[None], None]]:
    """
    Constructs callable functions corresponding to input configurations for parallelized decision-impact based sensing
    and compound score based moving decisions.

    Before sensing, all possible moves are scored on each board by Stockfish with a set of heuristics for evaluating
    board states unique to RBC, then each move is ranked based on a weighted-average score and on best- and worst-case
    scores among possible boards. Move scores are computed for sub-sets of boards corresponding to each possible sense
    result, and sensing choices are made to maximize the expected change in move scores before and after the sense
    result. Move scores are re-computed based on the observed sense result and the highest-scored move is made.
    Additionally, both sense and move strategies have small score incentives to reduce the set of possible boards. When
    time does not allow all possible boards to be evaluated, a random sample is taken.

    :param move_config: A dataclass of parameters which determine the move strategy's compound score
    :param score_config: A dataclass of parameters which determine the centi-pawn score assigned to a board's strength
    :param time_config: A dataclass of parameters which determine how time is allocated between turns

    :param board_weight_90th_percentile: The centi-pawn score associated with a 0.9 weight in the board set
    :param boards_per_centipawn: The scaling factor for combining decision-impact and set-reduction sensing

    :param num_workers: The number of StockFish engines to create for scoring moves
    :param num_threads: The number of threads for StockFish engine configuration (config skipped if None)

    :param checkmate_sense_override: A bool which toggles the corner-case sensing strategy for winning from checkmate
    :param while_we_wait_extension: A bool that toggles the scoring of boards that could be reached two turns ahead

    :param load_cache_data: A bool that tells whether to "warm up" the cache from a file of pre-calculated scores
    :param rc_disable_pbar: A bool which turns off tqdm progress bars if True

    :return: A tuple of callable functions (sense, move, ponder, exit)
    """

    logger = create_sub_logger('multiprocessing_strategies')
    logger.debug('Creating new instance of multiprocessing strategies.')

    # Initialize a list to store calculation time data for dynamic time management
    score_calc_times = []

    # Estimate calculation time based on data stored so far this game (and a provided starting datum)
    def calc_time_per_move() -> float:
        n0 = 100
        t0 = time_config.calc_time_per_move * n0
        total_num = n0 + sum(n for n, t in score_calc_times)
        total_time = t0 + sum(t for n, t in score_calc_times)
        return total_time / total_num

    # Determine how much of the remaining time should be spent on (the rest of) the current turn.
    def allocate_time(seconds_left: float, fraction_turn_passed: float = 0):
        turns_left = time_config.turns_to_plan_for - fraction_turn_passed  # account for previous parts of turn
        equal_time_split = seconds_left / turns_left
        return max(equal_time_split, time_config.min_time_for_turn)

    # Convert a board strength score into a probability for use in weighted averages (here using the logistic function)
    def weight_board_probability(score):
        return 1 / (1 + np.exp(-2 * np.log(3) / board_weight_90th_percentile * score))

    # If requested, pre-load the board/move score cache from a file
    if load_cache_data:
        logger.debug('Loading cached scores from file.')
        with open('Fianchetto/score_cache.json', 'r') as file:
            score_data = json.load(file)
        score_cache = score_data['cache']
        boards_in_cache = set(score_data['boards'])
    else:
        score_cache = dict()
        boards_in_cache = set()

    # Create the multiprocessing queues for communication with multiple StockFish engines
    request_queue = mp.Queue()
    response_queue = mp.Queue()

    # Memoized calculation of the score associated with one move on one board
    def memo_calc_score(board: chess.Board, moves: List[chess.Move]):
        key = make_cache_key(board)
        if key in score_cache:
            return score_cache[key]

        if not moves:
            moves = move_actions(board)
            if not chess.Move.null() in moves:
                moves.append(chess.Move.null())
        request_queue.put((key, moves))
        return None

    # Handler for requested scores. Filters for unique requests, then gets cached or calculated results.
    def memo_calc_set(requests):

        filtered_requests = {board: moves
                             for board, moves in requests}

        start = time()

        results = {make_cache_key(board):
                   memo_calc_score(board, moves)
                   for board, moves in filtered_requests.items()}

        num_new = sum(1 for score in results.values() if score is None)

        while any(score is None for score in results.values()):
            response = response_queue.get()
            # print('response:', response)
            score_cache.update(response)
            results.update(response)

        duration = time() - start
        if num_new:
            score_calc_times.append((num_new, duration))

        final_results = {}
        for board_fen, response in results.items():
            q, s, m = response
            asked_moves = filtered_requests[chess.Board(board_fen)]
            if asked_moves:
                if set(asked_moves) != set(m):
                    print(colored('Error: Asked moves different', 'red'))
                    exit()
                if asked_moves != m:
                    print(colored('Note: Asked moves ordered different', 'green'))
                    mapping = {i:m.index(move) for i, move in enumerate(asked_moves)}
                    # p = [p[mapping[i]] for i in range(len(p))]
                    s = [s[mapping[i]] for i in range(len(s))]

                    final_results[board_fen] = (q, s)
                else:
                    final_results[board_fen] = response[:-1]
            else:
                final_results[board_fen] = response[:-1]

        return final_results

    # Add a new board to the cache (evaluate the board's strength and relative score for every possible move).
    def cache_board(board: chess.Board):
        # board.turn = not board.turn
        op_score = memo_calc_set([(board, None)])[make_cache_key(board)]
        # board.turn = not board.turn
        boards_in_cache.add(board.fen())
        # memo_calc_set([(board, move, -op_score) for move in generate_moves_without_opponent_pieces(board)])

    # Randomly sample from the board set, but also include all of the boards which are already in the cache.
    def cache_favored_random_sample(board_set: Set[str], sample_size):
        prescored_boards = board_set & boards_in_cache
        return list(prescored_boards) + random.sample(board_set - prescored_boards,
                                                      min(len(board_set) - len(prescored_boards), sample_size))

    # Randomly choose one board from the board set, excluding boards which are already in the cache.
    def choose_uncached_board(board_set: Set[str]):
        uncached_boards = board_set - boards_in_cache
        return random.choice(tuple(uncached_boards)) if uncached_boards else None

    # Create and start the requested number of StockFish "worker" processes
    workers = [mp.Process(target=worker, args=(request_queue, response_queue, score_config, num_threads)) for _ in range(num_workers)]
    for process in workers:
        process.start()

    BOARD_SET_LIMIT = 1_000_000

    BSR_sense_x1=500
    BSR_sense_y1=1
    BSR_sense_x2=BOARD_SET_LIMIT
    BSR_sense_y2=100

    BSR_move_x1=500
    BSR_move_y1=1
    BSR_move_x2=BOARD_SET_LIMIT
    BSR_move_y2=100

    def BSR_factor_sense(board_set_size):
        if board_set_size<=BSR_sense_x1:
            return BSR_sense_y1
        else:
            return BSR_sense_y1+((BSR_sense_y2-BSR_sense_y1)*((board_set_size-BSR_sense_x1)/(BSR_sense_x2-BSR_sense_x1)))

    def BSR_factor_move(board_set_size):
        if board_set_size<=BSR_move_x1:
            return BSR_move_y1
        else:
            return BSR_move_y1+((BSR_move_y2-BSR_move_y1)*((board_set_size-BSR_move_x1)/(BSR_move_x2-BSR_move_x1)))

    def sense_strategy(board_set: Set[str], our_color: bool,
                       sense_actions: List[Square], moves: List[chess.Move],
                       seconds_left: float):
        """
        Choose a sense square to maximize the expected effect on move scores (to best inform the next move decision).

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. All possible moves are scored on these boards,
        and the combinations of scores for each possible sense result (since each sense would validate/invalidate
        different boards) are calculated. The sense square is chosen to maximize the change in move scores from before
        to after the sense.

        Centipawn points are also added per board for an expected board set size reduction by a sense choice. If the
        board set size is large enough, this becomes the dominant decision-making influence.

        Finally, a corner case is added to pinpoint the opponent's king in cases where we are (nearly) sure that their
        king is in check on all possible boards.
        """
        global msi, esr
#        print('movelist')
#        print(moves)
        # Don't sense if there is nothing to learn from it
        if len(board_set) == 1:
            return None

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = allocate_time(seconds_left)
        time_for_phase = time_for_turn * time_config.time_for_sense
        time_per_move = calc_time_per_move()
        time_per_board = time_per_move #* len(moves)
        sample_size = max(num_workers, int(time_for_phase / time_per_board))

        logger.debug('In sense phase with %.2f seconds left. Allocating %.2f seconds for this turn and %.2f seconds '
                     'for this sense step. Estimating %.4f seconds per calc over %d moves is %.4f seconds per '
                     'board score so we have time for %d boards.',
                     seconds_left, time_for_turn, time_for_phase, time_per_move,
                     len(moves), time_per_board, sample_size)

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        total_weighted_probability = 0
        sense_results = defaultdict(lambda: defaultdict(set))
        sense_possibilities = defaultdict(set)
        king_locations = defaultdict(lambda: defaultdict(set))

        # Get a random sampling of boards from the board set
        board_sample = cache_favored_random_sample(board_set, sample_size)
        # Initialize arrays for board and move data (dictionaries work here, too, but arrays were faster)
        board_sample_weights = np.zeros(len(board_sample))
        move_scores = np.zeros([len(moves), len(board_sample)])

        logger.debug('Sampled %d boards out of %d for sensing.', len(board_sample), len(board_set))

        # Get board position strengths before move for all boards in sample (to take advantage of parallel processing)
        board_score_reqs = []
        for board_epd in board_sample:
            board = chess.Board(board_epd)
            board.turn = our_color
            board_score_reqs.append((board, moves))
        board_score_dict = memo_calc_set(board_score_reqs)

        # board_sample_weights = softmax([-(board_score_dict[make_cache_key(board)][0]) for board in board_sample])

        for num_board, board_epd in enumerate(tqdm(board_sample, disable=rc_disable_pbar,
                                                   desc=f'{chess.COLOR_NAMES[our_color]} '
                                                        'Calculating choose_sense scores '
                                                        f'{len(moves)} moves in {len(board_set)} boards',
                                                   unit='boards')):
            board = chess.Board(board_epd)

            board.turn = our_color
            # print('results:', board_score_dict[make_cache_key(board)])
            my_score, all_moves_score = board_score_dict[make_cache_key(board)]
            op_score = -my_score #assumption
            board_sample_weights[num_board] = weight_board_probability(op_score)
            total_weighted_probability += board_sample_weights[num_board]

            # board.turn = our_color
            boards_in_cache.add(board.fen())  # Record that this board (and all moves) are in our cache

            # move_score_dict = get_move_prob(board, moves)  # Score all moves

            # Place move scores into array for later logical indexing
            move_scores[:, num_board] = all_moves_score

            # Gather information about sense results for each square on each board (and king locations)
            for square in SEARCH_SPOTS:
                sense_result = simulate_sense(board, square)
                num_occurances[square][sense_result] += 1
                weighted_probability[square][sense_result] += board_sample_weights[num_board]
                sense_results[board_epd][square] = sense_result
                sense_possibilities[square].add(sense_result)
                king_locations[square][sense_result].add(board.king(not our_color))

        # Take a different strategy if we are sure they are in checkmate (the usual board weight math fails there)
        if checkmate_sense_override and \
                all(board_sample_weights == weight_board_probability(score_config.into_check_score)):
            logger.debug("All scores indicate checkmate, therefore sensing based on king location.")
            num_king_squares = {square: np.mean([len(n) for n in king_locations[square].values()])
                                for square in SEARCH_SPOTS}
            min_num_king_squares = min(num_king_squares.values())
            sense_choice = random.choice([square for square, n in num_king_squares.items()
                                          if n == min_num_king_squares])
            return sense_choice

        # Calculate the mean, min, and max scores for each move across the board set (or at least the random sample)
        full_set_mean_scores = (np.average(move_scores, axis=1, weights=board_sample_weights))
        full_set_min_scores = (np.min(move_scores, axis=1))
        full_set_max_scores = (np.max(move_scores, axis=1))

        # Find the expected change in move scores caused by any sense choice
        sense_impact = defaultdict(lambda: defaultdict(float))
        for square in tqdm(SEARCH_SPOTS, disable=rc_disable_pbar,
                           desc=f'{chess.COLOR_NAMES[our_color]} Evaluating sense impacts '
                                f'for {len(board_set)} boards', unit='squares'):
            possible_results = sense_possibilities[square]
            for sense_result in possible_results:
                if len(possible_results) > 1:
                    subset_index = [i for i, board_epd in enumerate(board_sample)
                                    if sense_result == sense_results[board_epd][square]]
                    subset_move_scores = move_scores[:, subset_index]
                    subset_board_weights = board_sample_weights[subset_index]

                    # Calculate the mean, min, and max scores for each move across the board sub-set
                    sub_set_mean_scores = (np.average(subset_move_scores, axis=1, weights=subset_board_weights))
                    sub_set_min_scores = (np.min(subset_move_scores, axis=1))
                    sub_set_max_scores = (np.max(subset_move_scores, axis=1))

                    # Subtract the full set scores from the sub-set scores (and take the absolute value)
                    change_in_mean_scores = np.abs(sub_set_mean_scores - full_set_mean_scores)
                    change_in_min_scores = np.abs(sub_set_min_scores - full_set_min_scores)
                    change_in_max_scores = np.abs(sub_set_max_scores - full_set_max_scores)

                    # Combine the mean, min, and max changes in scores based on the config settings
                    change_in_compound_score = (
                        change_in_mean_scores * move_config.mean_score_factor +
                        change_in_min_scores * move_config.min_score_factor +
                        change_in_max_scores * move_config.max_score_factor
                    )

                    # The impact of this sense result is the resulting average change in absolute value of move scores
                    sense_impact[square][sense_result] = float(np.mean(change_in_compound_score))

                else:
                    sense_impact[square][sense_result] = 0

        # Calculate the expected mean change in centipawn score for each sense square
        mean_sense_impact = {
            square:
                sum([sense_impact[square][sense_result] * weighted_probability[square][sense_result]
                     for sense_result in sense_possibilities[square]]) / total_weighted_probability
            for square in SEARCH_SPOTS
        }

        # Also calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(board_set) *
                (1 - (1 / len(board_sample) / total_weighted_probability) *
                 sum([num_occurances[square][sense_result] * weighted_probability[square][sense_result]
                      for sense_result in sense_possibilities[square]]))
            for square in SEARCH_SPOTS
        }

        # Combine the decision-impact and set-reduction estimates
        for square in SEARCH_SPOTS:
            msi.append(mean_sense_impact[square])
            esr.append(expected_set_reduction[square])
        sense_score = {square:
                       mean_sense_impact[square] + (BSR_factor_sense(len(board_set)))*(expected_set_reduction[square] / boards_per_centipawn)
                       # mean_sense_impact[square] + 1.5*expected_set_reduction[square]
                       for square in SEARCH_SPOTS}

        max_sense_score = max(sense_score.values())
        sense_choice = random.choice([square for square, score in sense_score.items()
                                      if abs(score - max_sense_score) < SCORE_ROUNDOFF])

        # print(f'got sense choice {sense_choice}')
        return sense_choice

    def move_strategy(board_set: Set[str], our_color: bool,
                      moves: List[chess.Move],
                      seconds_left: float):
        """
        Choose the move with the maximum score calculated from a combination of mean, min, and max possibilities.

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. Each move is scored on each board, and the
        resulting scores are assessed together by looking at the worst-case score, the average score, and the best-case
        score. The relative contributions of these components to the compound score are determined by a config object.
        If requested by the config, bonus points are awarded to moves based on the expected number of boards removed
        from the possible set by attempting that move. Deterministic move patterns are reduced by randomly choosing a
        move that is within a few percent of the maximum score.
        """
        global msr
        global move_ind
        global store
        move_ind += 1
        store.append({})
        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = allocate_time(seconds_left, fraction_turn_passed=1-time_config.time_for_move)
        time_for_phase = time_for_turn * time_config.time_for_move
        time_per_move = calc_time_per_move()
        time_per_board = time_per_move #* len(moves)
        sample_size = max(1, int(time_for_phase / time_per_board))

        logger.debug('In move phase with %.2f seconds left. Allowing up to %.2f seconds for this move step. '
                     'Estimating %.4f seconds per calc over %d moves is %.4f seconds per '
                     'board score so we have time for %d boards.',
                     seconds_left, time_for_turn, time_per_move,
                     len(moves), time_per_board, sample_size)

        move_scores = defaultdict(list)
        weighted_sum_move_scores = defaultdict(float)

        # Initialize some parameters for tracking information about possible move results
        num_occurances = defaultdict(lambda: defaultdict(int))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        move_possibilities = defaultdict(set)
        total_weighted_probability = 0

        # Get a random sampling of boards from the board set
        board_sample = cache_favored_random_sample(board_set, sample_size)
        print('Sampled %d boards out of %d for moving.', len(board_sample), len(board_set))

        # Get board position strengths before move for all boards in sample (to take advantage of parallel processing)
        board_score_reqs = []
        for board_epd in board_sample:
            board = chess.Board(board_epd)
            board.turn = our_color
            board_score_reqs.append((board, moves))
            # oppboard = chess.Board(board_epd)
            # oppboard.turn = not our_color
            # board_score_reqs.append((oppboard, None))
        board_score_dict = memo_calc_set(board_score_reqs)

        # def check_score_wrapper(board):
        #     board = chess.Board(board)
        #     board.turn = our_color
        #     if board.is_check():
        #         return 1
        #     else:
        #         return -(board_score_dict[make_cache_key(board)][0])
        # board_sample_weights = softmax([check_score_wrapper(board) for board in board_sample])


        for num_board, board_epd in enumerate(tqdm(board_sample, disable=rc_disable_pbar,
                              desc=f'{chess.COLOR_NAMES[our_color]} Calculating choose_move scores '
                                   f'{len(moves)} moves in {len(board_set)} boards', unit='boards')):
            board = chess.Board(board_epd)

            board.turn = our_color
            my_score, all_moves_score = board_score_dict[make_cache_key(board)]
            op_score = -my_score
            board_weight = weight_board_probability(op_score)
            total_weighted_probability += board_weight

            # board.turn = our_color
            boards_in_cache.add(board.fen())  # Record that this board (and all moves) are in our cache

            # move_score_dict = get_move_prob(board, moves)  # Score all moves

            # Place move scores into array for later logical indexing
            # move_scores[:, num_board] = all_moves_prob

            # Gather scores and information about move results for each requested move on each board
            for it, move in enumerate(moves):
                score = all_moves_score[it]
                move_scores[move].append(score)
                weighted_sum_move_scores[move] += score * board_weight

                sim_move = simulate_move(board, move) or chess.Move.null()
                move_result = (sim_move, board.is_capture(sim_move))
                move_possibilities[move].add(move_result)
                num_occurances[move][move_result] += 1
                weighted_probability[move][move_result] += board_weight

        # Combine the mean, min, and max possible scores based on config settings
        compound_score = {move: (
            weighted_sum_move_scores[move] / total_weighted_probability * move_config.mean_score_factor +
            min(scores) * move_config.min_score_factor +
            max(scores) * move_config.max_score_factor
        ) for (move, scores) in move_scores.items()}


        store[-1]['#boards'] = len(board_set)
        store[-1]['#sample'] = len(board_sample)

        # Add centipawn points to a move based on an estimate of the board set reduction caused by that move
        # for move in compound_score:
        #     msr.append(1 / boards_per_centipawn *
        #                       (1 - (1 / len(board_sample)) *
        #                        sum([num_occurances[move][move_result] * weighted_probability[move][move_result]
        #                             for move_result in move_possibilities[move]])))
        # if move_config.sense_by_move:
        #     compound_score = {move: score + (BSR_factor_move(len(board_set)))*1 / boards_per_centipawn *
        #                       (1 - (1 / len(board_sample)) *
        #                        sum([num_occurances[move][move_result] * weighted_probability[move][move_result]
        #                             for move_result in move_possibilities[move]]))
        #                       for move, score in compound_score.items()}

        # Determine the minimum score a move needs to be considered
        highest_score = max(compound_score.values())
        threshold_score = highest_score - abs(highest_score) * move_config.threshold_score_factor
        # print(highest_score, threshold_score)
        # threshold_score = highest_score * move_config.threshold_score_factor


        # Create a list of all moves which scored above the threshold
        store[-1]['compound_score'] = {k:v for k, v in compound_score.items() if v >= threshold_score}
        move_options = [move for move, score in compound_score.items() if score >= threshold_score]
        # print(store[-1]['compound_score'])
        # # Eliminate move options which we know to be illegal (mainly for replay clarity)
        # move_options = [move for move in move_options
        #                 if move in {taken_move for taken_move, _ in move_possibilities[move]}]
        # Randomly choose one of the remaining moves
        try:
            move_choice = random.choice(move_options)
            # print('move choice:', move_choice)
        except Exception as e:
            print(e)
            print(len(board_set))
            if len(board_set) == 1:
                print(list(board_set)[0])
            print('compound_score:', compound_score)
            print('highest score:', highest_score)
            print('threshold_score:', threshold_score)
            print('move_options:', move_options)
            print('move_possibilities')
            print(move_possibilities)

        ############################################
        # assert abs(sum(compound_score.values()) - 1) < 1e-5, f"Move probs summing to {sum(compound_score.values())}"
        # move_options = sorted(compound_score.items(), key=lambda x: x[1], reverse=True)[:max(1, int(len(compound_score)/10))]
        # move_options = list(zip(*move_options))
        # move_choice = random.choices(move_options[0], weights=move_options[1])[0]

        return force_promotion_to_queen(move_choice) if move_config.force_promotion_queen else move_choice

    def while_we_wait(board_set: Set[str], our_color: bool):
        """
        Calculate scores for moves on next turn's boards. Store to cache for later processing acceleration.
        """
        print('www starts')
        if board_set:
            uncached_board_epd = choose_uncached_board(board_set)

            # If there are still boards for next turn without scores calculated, calculate move scores for one
            if uncached_board_epd:
                board = chess.Board(uncached_board_epd)
                print('caching uncached')
                cache_board(board)

            # Otherwise, calculate move scores for a random board that could be reached in two turns
            elif while_we_wait_extension:
                print('starting 2 moves lookahead')
                board = chess.Board(random.choice(tuple(board_set)))
                board.push(random.choice(list(generate_rbc_moves(board))))
                board.push(random.choice(list(generate_rbc_moves(board))))
                if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
                    cache_board(board)

            else:
                sleep(0.001)

    def end_game():
        """
        Quit the StockFish engine instance(s) associated with this strategy once the game is over.
        """
        global msi
        global esr
        global msr
        global store
        # np.save('qarr.npy', Qs)
        np.save('msi', msi)
        np.save('esr', esr)
        np.save('msr', msr)
        np.save('store', store)
        logger.debug("During this game, averaged %.5f seconds per score using search depth %d and %d workers.",
                     calc_time_per_move(), score_config.search_depth, num_workers)

        # Shut down the StockFish "workers"
        [process.terminate() for process in workers]
        [process.join() for process in workers]

    # Return the callable functions so they can be used by StrangeFish
    return sense_strategy, move_strategy, while_we_wait, end_game
