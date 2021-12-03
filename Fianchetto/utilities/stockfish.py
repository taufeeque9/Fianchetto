import os

import chess.engine
from lczero.backends import Weights, Backend, GameState, LczeroException
# make sure stockfish environment variable exists
if "STOCKFISH_EXECUTABLE" not in os.environ:
    raise KeyError('This bot requires an environment variable called "STOCKFISH_EXECUTABLE"'
                   ' pointing to the Stockfish executable')
# make sure there is actually a file
STOCKFISH_EXECUTABLE = os.getenv('STOCKFISH_EXECUTABLE')
if not os.path.exists(STOCKFISH_EXECUTABLE):
    raise ValueError('No stockfish executable found at "{}"'.format(STOCKFISH_EXECUTABLE))


def create_engine(max_batch, gpu_id):
    print("MAX_BATCH SIZE - ",max_batch)
    w = Weights('weights_run3_752050.pb.gz')
    # w = Weights('LS15_20x256SE_jj_9_75000000.pb.gz')
#    w = Weights('weights_run1_610024.pb.gz')
    try:
        b = Backend(weights=w, backend='cuda-fp16', options=f'max_batch={max_batch}, gpu={gpu_id}')
    except LczeroException:
        b = Backend(weights=w, backend='cuda', options=f'max_batch={max_batch}, gpu={gpu_id}')
    return b
