
import numpy as np
from activation import sigmoid, sigmoidPrime

class N_Network(object):
  def __init__(self):
    # the size of each node in each layer
    self.inputSize = 14
    self.outputSize = 1
    self.hiddenSize = 9

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (14x9) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (9x1) weight matrix from hidden to output layer

  def forward_propagation(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 14x9 weights
    self.z2 = sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 9x1 weights
    output= sigmoid(self.z3) # final activation function
    return output

  def backward_propagation(self, X, y, output):#backward propgate through the network
    self.output_error = np.subtract(y, output) #y - o # error in output
    self.output_delta = self.output_error*sigmoidPrime(output) # applying derivative of sigmoid to error

    self.z2_error = self.output_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    output = self.forward_propagation(X)
    self.backward_propagation(X, y, output)

  def save_updated_Weights(self):#saved updated weights
    np.savetxt("updated_w1.txt", self.W1, fmt="%s")
    np.savetxt("updated_w2.txt", self.W2, fmt="%s")

  def make_prediction(self, xPredicted):# predicts the test score of the next exam
    print("Predicted data based on updated weights: ");
    print("Input (scaled): \n" + str(xPredicted));
    print("predicted Output of the next test score: \n" + str(self.forward_propagation(xPredicted)));
