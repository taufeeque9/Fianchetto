from dataclasses import dataclass
from typing import List, Set, Callable, Tuple
import warnings
import chess
import chess.engine
from reconchess import is_psuedo_legal_castle
from reconchess.utilities import capture_square_of_move

from Fianchetto.utilities import simulate_move
import numpy as np

from lczero.backends import Weights, Backend, GameState
from scipy.special import softmax as softmax
from termcolor import colored
warnings.simplefilter("error", RuntimeWarning)
# get better sneak rewards
@dataclass
class ScoreConfig:
    capture_king_score: float = 200  # bonus points for a winning move
    checkmate_score: int = 120  # point value of checkmate
    stalemate_score: int = 80  # point value of stalemate
    into_check_score: float = -160  # point penalty for moving into check
    search_depth: int = 8  # Stockfish engine search ply
    reward_attacker: float = 1.5  # Bonus points if move sets up attack on enemy king
    require_sneak: bool = True  # Only reward bonus points to aggressive moves if they are sneaky (aren't captures)
    far_away_defense_score: float = 0.5 #Bonus points for protecting against check from far away with support (fads)


def map_uci(uci: str, board: chess.Board = None):
    """Returns standard UCI moves (including castling) when board is also given"""
    if board and board.piece_at(chess.parse_square(uci[:2])).piece_type is chess.KING:
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

def calculate_score(b,
                    board_fen,
                    moves: List[chess.Move],
                    score_config: ScoreConfig = ScoreConfig()):

    # pov = board.turn

    # if next_board.was_into_check():
    #     return score_config.into_check_score
    board = chess.Board(board_fen)
    pov = board.turn
    game = GameState(fen=board_fen)
    inp = game.as_input(b)
    out = b.evaluate(inp)[0]
    q = out.q()
    raw_scores_c = {map_uci(uci, board): score for uci, score in zip(game.moves(), out.p_raw(*game.policy_indices()))}

    if len(raw_scores_c) > 0:
        least_score = min(raw_scores_c.values())
    else:
        raw_scores_rbc = [0]*len(moves)
        raw_scores_rbc[moves.index(chess.Move.null())] = 10
        print(colored(f'No moves in this board (q = {q}):', 'red'))
        print(board_fen)
        return q, raw_scores_rbc, moves

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
                if board.piece_at(capture_square_of_move(board, move)).piece_type is chess.KING:
                    raw_scores_rbc[i] = score_config.capture_king_score
                    q = 1
                    break
        elif move == chess.Move.null():
            if board.is_check():
                raw_scores_rbc[i] = score_config.into_check_score
            else:
                raw_scores_rbc[i] = -2*abs(least_score)
            continue

        next_board = board.copy()
        next_board.push(move)
        next_board.clear_stack()
        if next_board.was_into_check():
            raw_scores_rbc[i] = score_config.into_check_score
        elif next_board.is_checkmate():
            raw_scores_rbc[i] = score_config.checkmate_score
        else:
            if board.was_into_check():                                                                            
                king_attackers = board.attackers(not pov, board.king(pov))                         # list of squares/pieces that attack our king
                for square in king_attackers:                                                      
                    if ((board.piece_type_at(square)==3) or (board.piece_type_at(square)==4) or (board.piece_type_at(square)==5)):  # 3->Bishop, 4->Rook, 5->Queen
                        support=next_board.attackers(pov,move.to_square)
                        opposition=next_board.attackers(not pov,move.to_square)
                        if len(list(support))>=len(list(opposition)):
                            if chess.square_distance(move.to_square,board.king(pov))>2:
                                raw_scores_rbc[i]=score_config.far_away_defense_score
                                break

            # print('getting from neural')
            if move in raw_scores_c:
                raw_scores_rbc[i] += raw_scores_c[move]

            if next_board.is_stalemate():
                raw_scores_rbc[i]+=score_config.stalemate_score

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
                if len(list(support))<len(list(opposition)):
                    if not score_config.require_sneak:  # and we don't require the attackers to be sneaky
                        raw_scores_rbc[i] -= score_config.reward_attacker/3  # subtract some bonus points
                    # or if we do require the attackers to be sneaky, either the last move was not a capture (which would give away
                    # our position) or there are now attackers other than the piece that moves (discovered check)
                    elif not next_board.is_capture(move) or any([square != move.to_square for square in king_attackers]):
                        raw_scores_rbc[i] -= score_config.reward_attacker/3  # subtract some bonus points


    for i, revised_move in revised.items():
        raw_scores_rbc[i] = raw_scores_rbc[revised_move]

    # board.push(chess.Move.null())
    # if board.was_into_check():
    #     q = -5
    # print(raw_scores_rbc)
    # if all(np.array(raw_scores_rbc) == np.NINF):
    #     raw_scores_rbc[moves.index(chess.Move.null())] = 0

    # try:
    #     p = softmax(raw_scores_rbc)
    # except RuntimeWarning as e:
    #     print(e)
    #     print('raw_scores_rbc in except:', raw_scores_rbc)
    #     print(board.fen())
    #     print('raw_score_c:',raw_scores_c)
    #     print('moves:', moves)

    return (q, raw_scores_rbc, moves)
