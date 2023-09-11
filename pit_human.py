import Arena
from MCTS import MCTS
from game.Players import *
from game.NNet import NNetWrapper as NNet
import numpy as np
import torch
import random
from Config import get_config
import numpy as np
from utils.utils import *
import sys

seed = int(sys.argv[1])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

game_num = 10
wins = {}

# Define first player
parser1 = get_config(1)
args1 = parser1.parse_args(sys.argv[2:])
args1.policy = "AOAP-Gaussian"
args1.StochasticAction = True
args1.sigmma_0 = 0.1
args1.exploreSteps = 1
n1 = NNet(args1.boardsize, args1)
n1.load_checkpoint('./checkpoint/Iter1/','best.pth.tar')
args1.numMCTSSims = 50
mcts1 = MCTS(n1, args1.policy, args1)
n1p = lambda x: mcts1.getActionProb(x)

# Define second player
parser2 = get_config(1)
args2 = parser2.parse_args(sys.argv[2:])
args2.policy = "UCT"
args2.StochasticAction = True
args2.cpuct = 1
args2.exploreSteps = 1
n2 = NNet(args2.boardsize, args2)
n2.load_checkpoint('./checkpoint/Iter1/','best.pth.tar')
args2.numMCTSSims = 50
mcts2 = MCTS(n2, args2.policy, args2)
n2p = lambda x: mcts2.getActionProb(x)

# Compete
arena = Arena.Arena(n1p, n2p, args1)
one_win, two_win, diffs = arena.playGames(game_num, verbose=True)
print(one_win, two_win, game_num - one_win - two_win)
