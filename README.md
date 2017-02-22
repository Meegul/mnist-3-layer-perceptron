Just clone, make sure Tensorflow is installed, and run:
`python loadModel.py` to run the trained model, or
`python trainModel.py` to train a new model.


This NN-configuration should get around 96.5% accuracy on the mnist dataset. It's a simple 3-layer perceptron, where each layer consists of 256 relu-activated neurons. The neural network uses softmax and cross entropy for its cost function.
