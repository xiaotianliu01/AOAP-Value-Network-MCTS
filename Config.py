import argparse
import torch

def get_config(IterNumber):

    parser = argparse.ArgumentParser(description='AlphaGo AOAP')

    parser.add_argument("--boardsize", type=int,
                        default=3, choices=[3, 8])
    # Size for board. Change this param to 8 and change the param "self.connection_length_for_win" in line 15 of file ./game/Game.py to 5, the game becomes Five-in-a-row.
    parser.add_argument("--board_feature_channel", type=int,
                        default=5, choices=[5, 9])
    # Number of historical boards to generate features
    parser.add_argument("--numEps", type=int,
                        default=20)
    # Number of games simulated for one model iteration
    parser.add_argument("--policy", type=str,
                        default='UCT', choices=['UCT', 'AOAP-Gaussian'])
    # Node selection policy in MCTS
    parser.add_argument("--StochasticAction", type=bool,
                        default=True, choices=[True, False])
    # Stochastic action or deterministic action
    parser.add_argument("--AOAP_Pi", type=bool,
                        default=False, choices=[True, False])
    # If use AOAP-Pi
    parser.add_argument("--numMCTSSims", type=int,
                        default=100)
    # Search budget for each MCTS
    parser.add_argument("--num-sim-processes", type=int,
                        default=10)
    # Number of processes to parrelly simulate games for one mode iteration 
    parser.add_argument("--cpuct", type=float,
                        default=1)
    # Param in UCT
    parser.add_argument("--sigmaa_0", type=float,
                        default=0.25)
    # Param in AOAP

    parser.add_argument("--checkpoint", type=str,
                        default='./checkpoint/Iter' + str(IterNumber) + '/')
    # Path to save simulation data and NN models
    parser.add_argument("--load_folder_file", type=tuple,
                        default=('./checkpoint/Iter' + str(IterNumber) + '/','nn.pth.tar'))
    # Path to load simulation data and NN models
    parser.add_argument("--load_nn", type=tuple,
                        default=('./checkpoint/Iter' + str(IterNumber-1) + '/','best.pth.tar'))
    # Path to load NN models as initialization
    parser.add_argument("--IterNumber", type=int,
                        default=IterNumber)
    # Number of iteration
    
    file_list = []
    if(IterNumber+1 <= 20):
        for i in range(1, IterNumber+1):
            file_list.append('./checkpoint/Iter' + str(i) + '/')
    else:
        for i in range(IterNumber+1-20, IterNumber+1):
            file_list.append('./checkpoint/Iter' + str(i) + '/')
    
    if(IterNumber+1 <= 15):
        exp = 2
    else:
        exp = 1
    
    parser.add_argument("--exploreSteps", type=int,
                        default=exp)
    # Exploration steps               
    parser.add_argument("--samples_pths", type=int,
                        default=file_list)
    # Paths to all the simulation data from multiple previous iterations that are used in current NN training
    parser.add_argument("--lr", type=float,
                        default=0.0005)
    # Learning rate
    parser.add_argument("--dropout", type=float,
                        default=0.3)
    # Dropout rate
    parser.add_argument("--epochs", type=int,
                        default=2)
    # Epochs for one iteration
    parser.add_argument("--batch_size", type=int,
                        default=64)
    # Batch size
    parser.add_argument("--cuda", type=bool,
                        default=torch.cuda.is_available())
    # Use GPU or CPU
    parser.add_argument("--num_channels", type=int,
                        default=512)
    # Size of the NN network
                        
    return parser
