import random

class Perceptron():
    def __init__ (self, maxEpochs = -1, learningRate = 0.3, bias = 0.5):
        self.maxEpochs = maxEpochs
        self.learningRate = learningRate
        self.bias = bias
        self.inputs = []
        self.targets = []
        self.weights = []

    def train(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.weights = self.initWeights(len(inputs[0]))

        #insert bias input
        for _input in range(0, len(inputs)):
            inputs[_input].insert(0, 1)

        epoch = 0
        #print (self.weights)
        if(self.maxEpochs == -1):
            converged = False
            while(converged == False):
                epoch += 1
                print(epoch)
                for nSample in range(0, len(inputs)):
                    #print(inputs[nSample])
                    scalarProduct = 0.0
                    #scalar product
                    for nInput in range(0, len(inputs[nSample])):
                        scalarProduct += self.inputs[nSample][nInput] * self.weights[nInput]

                    #set the output from the perceptron
                    output = self.thresholdFunc(scalarProduct)

                    #update the weights
                    self.weights = self.updateWeights(self.inputs[nSample], self.weights, self.targets[nSample], output)

                #verify if it converged
                converged = True

                for nSample in range(0, len(inputs)):
                    if(targets[nSample] != self.fit(inputs[nSample][1:])):
                        converged = False
                        break
        else:
            for epoch in range(0, self.maxEpochs):
                #iterate each sample
                for nSample in range(0, len(inputs)):
                    scalarProduct = 0.0

                    #scalar product
                    for nInput in range(0, len(inputs[nSample])):
                        scalarProduct += self.inputs[nSample][nInput] * self.weights[nInput]

                    #set the output from the perceptron
                    output = self.thresholdFunc(scalarProduct)

                    #update the weights
                    self.weights = self.updateWeights(self.inputs[nSample], self.weights, self.targets[nSample], output)

    def initWeights(self, nWeights):
        return [self.bias, random.random(), random.random()]

    def thresholdFunc(self, _y):
        if(_y >= 0):
            return 1
        else:
            return -1

    def updateWeights(self, inputs, weights, target, output):
        #print(weights)
        for nWeight in range(0, len(weights)):
            weights[nWeight] += self.learningRate * (target - output) * inputs[nWeight]
        #print(weights)
        return weights

    def fit(self, inputs):
        if(len(self.weights) == 0):
            return "Train the perceptron please."

        scalarProduct = 0.0

        inputs.insert(0, 1)
        for i in range(0, len(inputs)):
            scalarProduct += inputs[i] * self.weights[i]

        #print("scalar: " + str(scalarProduct))
        output = self.thresholdFunc(scalarProduct)

        return output
