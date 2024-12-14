# Federalisation example

This project shows the process of turning sequential code into a federated learning application through the use of a federeated learning development paradigm outlined [here](https://arxiv.org/abs/2310.05102).

Baseline code is outlined [here](https://www.kaggle.com/code/alokevil/non-linear-regression) along with the dataset. Regression problem outlined on this page was solved using linear regression with the help of the [scikit-learn](https://scikit-learn.org/1.5/index.html) library.

This project in its later stages takes a different approach using a Feed-forward neural network written in [PyTorch](https://pytorch.org) to solve the given regression problem. The network consists of three hidden layers with 100 units per layer, LeakyRelu activation functions and a dropout of 0.01.

After acheiving similar performance as the baseline code, previously mentioned development paradigm was applied on the PyTorch/NN based aproach and as a result we got a federated learning application that uses the [PTB-FLA](https://github.com/miroslav-popovic/ptbfla?tab=readme-ov-file) federated learning framework.

This example therefore represents a showcase of the possible synergy between PTB-FLA federeated learning framework and other tools focused on machine learning such as PyTorch. 
