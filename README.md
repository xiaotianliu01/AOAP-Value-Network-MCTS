# AOAP-Value-Network-MCTS
Official code for ["An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4450999)

## Create Environment
The codes are implementable on Ubuntu with Python 3.6.5 and pytorch.

## How To Run
The main logic of the training process is shown in the following figure
<img src="https://github.com/xiaotianliu01/AOAP-Value-Network-MCTS/blob/master/diagram.png" width="400" height="300">

For each iteration, the python file *Simulate.py* is used to simulate the games to collect training data, and the python file *Learn.py* is used to train the NN models with the collected training data for one iteration.

***To collect the training data for one iteration, run***
```Bash
python Simulate.py 1
```
where the argument 1 is the number of iteration.

***To train the NN models for one iteration, run***
```Bash
python Learn.py 1
```
where the argument 1 is the number of iteration.

***You can also use the script *train.sh* to automatically do the training for multiple iterations***
```Bash
sh train.sh
```
where you can specify the number of iterations by modifying the file *train.sh*.

All exgeneous parameters are determined in the file *config.py*.

***To test the model, run***
```Bash
python pit_human.py 1
```
where the argument 1 is the random seed.

