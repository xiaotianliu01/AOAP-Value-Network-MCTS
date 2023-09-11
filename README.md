# AOAP-Value-Network-MCTS
Official code for ["An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4450999)

## Create Environment
The codes are implementable on Ubuntu with Python 3.6.5 and pytorch.

## How To Run
The main logic of the training process is shown in the following figure
![image](https://github.com/xiaotianliu01/AOAP-Value-Network-MCTS/blob/master/diagram.png){width=400px height=300px}

For each iteration, the python file *Simulate.py* is used to simulate the games to collect training data, and the python file *Learn.py* is used to train the NN models with the collected training data for one iteration.


