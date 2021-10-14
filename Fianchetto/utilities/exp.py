import os
from reconchess import GameHistory
import chess.engine
import chess
from lczero.backends import Weights, Backend, GameState
import time
# from .rbc_move_score import

STOCKFISH_EXECUTABLE = './stockfish_14_x64'


def create_engine():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_EXECUTABLE)
    # engine.configure({'Threads': os.cpu_count()})
    return engine

def stockfish(all_fen):
    engine = create_engine()
    count = 1
    for fen in all_fen:
        board = chess.Board(fen)
        try:
            engine_result = engine.analyse(board, chess.engine.Limit(depth=8))
            # print(engine_result)
            score = engine_result['score'].pov(board.turn).score(mate_score=30_000)
        except chess.engine.EngineTerminatedError:
            print(count)
            count += 1
            engine = create_engine()

    engine.quit()

w = Weights('weights_run1_609973.pb.gz')
b = Backend(weights=w, backend='opencl')

def lc0(all_fen):
    global w
    global b
    for fen in all_fen:
        game = GameState(fen=fen)
        inp = game.as_input(b)
        out = b.evaluate(inp)[0]
        best_moves = sorted(zip(game.moves(), out.p_softmax(*game.policy_indices())), key=lambda x: -x[1])


def main():
    history = GameHistory.from_file('game.json')
    all_fen = []
    for turn in history.turns():
        if history.has_move(turn):
            all_fen.append(history.truth_fen_after_move(turn))

    t0 = time.time()
    stockfish(all_fen)
    print('Stockfish time: ', time.time()-t0)
    t0 = time.time()
    lc0(all_fen)
    print('lc0 time: ', time.time()-t0)

if __name__ == "__main__":
    main()
