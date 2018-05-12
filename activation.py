
import numpy as np

def sigmoid(s):# activation function
     return 1/(1+np.exp(-s))

def sigmoidPrime(x):#derivative of sigmoid
     return x * (1 - x)
