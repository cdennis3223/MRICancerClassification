#HW Assignment 5, Example of how to build a simple NN
#Hyperparameters: 2 inputs, 3 neuron HL, 1 output
#Carl Dennis SID:007968429

import numpy as np
from scipy import optimize

class NeuralNetwork:
    def __init__(self):
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        # Initialize weights randomly
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        output = self.z2
        return output

    def costFunction(self, X, y):
        self.output = self.forward(X)
        cost = np.sum((y - self.output) ** 2) / len(y)
        return cost


    def costFunctionPrime(self, X, y):
        self.output = self.forward(X)
        delta2 = -(y - self.output)
        dJdW2 = np.dot(self.a1.T, delta2)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dJdW1 = np.dot(X.T, delta1)
        return dJdW1, dJdW2
    
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
    def getParams(self):
            params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
            return params
        
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hidden_size * self.input_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_size, self.hidden_size))
        W2_end = W1_end + self.hidden_size * self.output_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_size, self.output_size))
    
class trainer(object):
    def __init__(self,N):
        self.N = N

    def callBackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):

        self.X = X
        self.y = y

        #make empty list to store costs
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac =True,\
                                  method ='BFGS', args=(X,y), options = options, callback = self.callBackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

# Example usage
if __name__ == "__main__":
    # Training data for problem 5
    X = np.array([[3, 5],
                  [5, 1],
                  [10, 2],
                  ])
    y = np.array([[15], [16.4], [18.6]])

    # Train the neural network
    NN = NeuralNetwork()
    T = trainer(NN)
    T.train(X, y)

    # Test the trained model
    TestX = np.array([[8,3]])
    TestY = np.array([19])

    output = NN.forward(TestX)
    print("Predicted output for [8, 3]:", output)
    print("Actual output for [8, 3]:", TestY)




