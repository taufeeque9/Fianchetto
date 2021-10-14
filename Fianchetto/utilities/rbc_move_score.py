from dataclasses import dataclass
from typing import List, Set, Callable, Tuple
import warnings
import chess
import chess.engine
from reconchess import is_psuedo_legal_castle
from reconchess.utilities import capture_square_of_move, move_actions
from Fianchetto.utilities import simulate_move
import numpy as np

from lczero.backends import Weights, Backend, GameState
from scipy.special import softmax as softmax
from termcolor import colored
from . import generate_moves_without_opponent_pieces as all_rbc_moves, generate_rbc_moves
warnings.simplefilter("error", RuntimeWarning)
# get better sneak rewards
@dataclass
class ScoreConfig:
    capture_king_score: float = 200  # bonus points for a winning move
    checkmate_score: int = 120  # point value of checkmate
    into_check_score: float = -160  # point penalty for moving into check
    search_depth: int = 8  # Stockfish engine search ply
    reward_attacker: float = 1.5  # Bonus points if move sets up attack on enemy king
    require_sneak: bool = True  # Only reward bonus points to aggressive moves if they are sneaky (aren't captures)
    far_away_defense_score: float = 0.5 #Bonus points for protecting against check from far away with support (fads)

def cp(Q):
    return 111.714640912 * np.tan(1.5620688421 * Q)

def map_uci(uci: str, board: chess.Board = None):
    """Returns standard UCI moves (including castling) when board is also given"""
    if board and board.piece_type_at(chess.parse_square(uci[:2])) is chess.KING:
        if uci == "e8h8":
            return map_uci("e8g8")
        elif uci == "e8a8":
            return map_uci("e8c8")
        elif uci == "e1h1":
            return map_uci("e1g1")
        elif uci == "e1a1":
            return map_uci("e1c1")
        else:
            return map_uci(uci)
    else:
        return chess.Move.from_uci(uci)

def evaluate_batch(b, batch):
    """
    batch = list of epds
    Return their (q, chess_score_for_all_moves)
    """
    # all_fens = [board_epd for board_epd in batch]
    all_games = [GameState(fen=board_epd) for board_epd in batch]
    batch_for_nn = [game.as_input(b) for game in all_games]
    output = b.evaluate(*batch_for_nn)

    return output


def helper_f(b, all_board_moves, score_cache, time_config):

    results = {}
    uneval_boards = []
    cached_asked_moves = {}
    chess_output_dict = {}
    for board_epd, asked_moves in all_board_moves.items():
        # key = make_cache_key(board)
        if board_epd in score_cache:
            results[board_epd] = score_cache[board_epd]
            cached_asked_moves[board_epd] = asked_moves
        else:
            uneval_boards.append(board_epd)
            if len(uneval_boards) == time_config.max_batch:
                chess_output = evaluate_batch(b, uneval_boards)
                for i in range(len(uneval_boards)):
                    chess_output_dict[uneval_boards[i]] = chess_output[i]
                uneval_boards = []

        null_board = chess.Board(board_epd)
        null_board.push(chess.Move.null())
        null_board.clear_stack()
        null_board_epd = null_board.epd(en_passant='xfen')
        if null_board_epd in score_cache:
            results[null_board_epd] = score_cache[null_board_epd]
            cached_asked_moves[null_board_epd] = list(generate_rbc_moves(null_board))
        else:
            uneval_boards.append(null_board_epd)
            if len(uneval_boards) == time_config.max_batch:
                chess_output = evaluate_batch(b, uneval_boards)
                for i in range(len(uneval_boards)):
                    chess_output_dict[uneval_boards[i]] = chess_output[i]
                uneval_boards = []

    if len(uneval_boards) > 0:
        chess_output = evaluate_batch(b, uneval_boards)
        for i in range(len(uneval_boards)):
            chess_output_dict[uneval_boards[i]] = chess_output[i]
        uneval_boards = []

    # if 2*len(all_board_moves) != len(chess_output_dict):
    #     with open('/home/taufeeque/errors.txt', 'a') as f:
    #         f.write(f'abm: {len(all_board_moves)}, cod: {len(chess_output_dict)}\n')

#    assert 2*len(all_board_moves) == len(chess_output_dict)
    return results, cached_asked_moves, chess_output_dict

def calculate_score(b,
                    board_moves_dict,
                    score_cache,
                    time_config,
                    score_config: ScoreConfig = ScoreConfig()):
    """
    b = lc0 backend
    board_moves_dict = dict of {board_epd: moves (can be None)} (board_epd can be in cache)
    score_cache = cache of all scored boards
    """
    results, cached_asked_moves, chess_output_dict = helper_f(b, board_moves_dict, score_cache, time_config)

    for board_epd, out in chess_output_dict.items():
        board = chess.Board(board_epd)
        moves = board_moves_dict.get(board_epd, None)
        if not moves:
            moves = list(generate_rbc_moves(board))
        pov = board.turn
        q = cp(out.q())
        game = GameState(fen=board_epd)
        raw_scores_c = {map_uci(uci, board): score for uci, score in zip(game.moves(), out.p_raw(*game.policy_indices()))}

        if len(raw_scores_c) > 0:
            least_score = min(raw_scores_c.values())
        else:
            raw_scores_rbc = [0]*len(moves)
            raw_scores_rbc[moves.index(chess.Move.null())] = 10
            results[board_epd] = (q, raw_scores_rbc, moves)
            # score_cache[board_epd] = (q, raw_scores_rbc, moves)
            continue

        raw_scores_rbc = [-2*abs(least_score)]*len(moves)
        revised = {}


        for i, move in enumerate(moves):
            # print(move)
            if move != chess.Move.null() and not is_psuedo_legal_castle(board, move):
                # print('first if')
                if not board.is_pseudo_legal(move):
                    # check for sliding move alternate results, and score accordingly
                    revised_move = simulate_move(board, move)
                    if revised_move is not None:
                        revised[i] = moves.index(revised_move)
                    else:
                        if board.is_check():
                            raw_scores_rbc[i] = score_config.into_check_score
                        else:
                            raw_scores_rbc[i] = -2*abs(least_score)
                    continue
                elif board.is_capture(move):
                    if board.piece_type_at(capture_square_of_move(board, move)) is chess.KING:
                        raw_scores_rbc[i] = score_config.capture_king_score
                        q = score_config.capture_king_score*50
                        break
            elif move == chess.Move.null():
                continue

            next_board = board.copy()
            next_board.push(move)
            next_board.clear_stack()
            if next_board.was_into_check():
                raw_scores_rbc[i] = score_config.into_check_score
            elif next_board.is_checkmate():
                raw_scores_rbc[i] = score_config.checkmate_score
            else:
                # print('getting from neural')
                raw_scores_rbc[i] = raw_scores_c.get(move, -2*abs(least_score))

                if board.is_check():
                    king_attackers = board.attackers(not pov, board.king(pov))                         # list of squares/pieces that attack our king
                    for square in king_attackers:
                        if ((board.piece_type_at(square)==chess.BISHOP) or (board.piece_type_at(square)==chess.ROOK) or (board.piece_type_at(square)==chess.QUEEN)):  # 3->Bishop, 4->Rook, 5->Queen
                            support=next_board.attackers(pov,move.to_square)
                            opposition=next_board.attackers(not pov,move.to_square)
                            if len(list(support))>=len(list(opposition)):
                                if chess.square_distance(move.to_square,board.king(pov))>2:
                                    raw_scores_rbc[i]+=score_config.far_away_defense_score
                                    break

                # Add bonus board position score if king is attacked
                king_attackers = next_board.attackers(pov, next_board.king(not pov))  # list of pieces that can reach the enemy king
                if king_attackers:  # if there are any such pieces...
                    if not score_config.require_sneak:  # and we don't require the attackers to be sneaky
                        raw_scores_rbc[i] += score_config.reward_attacker  # add the bonus points
                    # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
                    # our position) or there are now attackers other than the piece that moves (discovered check)
                    elif not next_board.is_capture(move) or any([square != move.to_square for square in king_attackers]):
                        raw_scores_rbc[i] += score_config.reward_attacker  # add the bonus points

                if len(king_attackers)==1:
                    square = king_attackers.pop()
                    support=next_board.attackers(pov,square)
                    opposition=next_board.attackers(not pov,square)
                    opposition_piece_list=set(next_board.piece_type_at(sq) for sq in list(opposition))
                    pkbr=set([chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK])
                    if (len(list(support))<len(list(opposition))) or ((next_board.piece_type_at(square) == chess.QUEEN) and (len(pkbr & opposition_piece_list)!=0)):
                        ratio = 3
                        if next_board.piece_type_at(square) == chess.QUEEN:
                            ratio = 1
                        elif next_board.piece_type_at(square) == chess.ROOK:
                            ratio = 1.1
                        elif next_board.piece_type_at(square) == chess.PAWN:
                            ratio = 5
                        if not score_config.require_sneak:  # and we don't require the attackers to be sneaky
                            raw_scores_rbc[i] -= score_config.reward_attacker/ratio  # subtract some bonus points
                        # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
                        # our position) or there are now attackers other than the piece that moves (discovered check)
                        elif not next_board.is_capture(move) or any([square != move.to_square for square in king_attackers]):
                            raw_scores_rbc[i] -= score_config.reward_attacker/ratio  # subtract some bonus points


        for i, revised_move in revised.items():
            raw_scores_rbc[i] = raw_scores_rbc[revised_move]

        #handle null move
        if board.is_check():
            raw_scores_rbc[moves.index(chess.Move.null())] = score_config.into_check_score
        # else:
        #     next_board = board.copy()
        #     next_board.push(chess.Move.null())
        #     next_board.clear_stack()
        #     try:
        #         null_q = chess_output_dict[next_board.epd(en_passant='xfen')].q()
        #         raw_scores_rbc[moves.index(chess.Move.null())] = null_q*max(raw_scores_rbc)/out.q()
        #     except Exception as e:
        #         pass



        results[board_epd] = (q, raw_scores_rbc, moves)
        # score_cache[board_epd] = (q, raw_scores_rbc, moves)

    return results, cached_asked_moves