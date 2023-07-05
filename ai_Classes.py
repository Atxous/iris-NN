import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    def forward(self, inputs):
        exponentiated = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        #subtract by max so it won't overflow (etc e^800, now it'll be e^0)
        #(which is 1. if another input is 799, then it'll be e^-1 which is less than 1)
        #also, the inputs are scaled. so if it's [2, 3, 5], it means the model
        #thinks 5 is the class and thus why it is a greater number
        self.outputs = exponentiated / np.sum(exponentiated, axis = 1, keepdims = True)
        #basically e^x / e^x + e^y + e^z (x and y and z are inputs)
        #so in the first line it becomes an array of the e^x values
        #then in the second line that e^x value is divided by the whole
        #array that contains all the e^x values regarding the other inputs
        #so that we get percentages.
        
class Softmax_Cross_Entropy:
    def __init__(self):
        self.softmax = Softmax()
        self.cce = Categorical_Cross_Entropy_Cost()
    
    def forward(self, inputs, real):
        self.softmax.forward(inputs)
        return self.cce.forward(self.softmax.outputs, real)
    
    def backward(self, real):
        self.dinputs = self.softmax.outputs.copy()
        #probabilities
        self.dinputs[range(len(real)), real] -= 1
        #index for correct probabilities and subtract by 100%
        self.dinputs = self.dinputs / len(real)
        #take the average to stay consistent with forward method
        #because we took avg in forward methods
        #look at MSE, we divided to take avg
        
        #essentially the -1 makes makes the derivatives positive/negative
        #what this means is that positive means it's wrong and negative means it's
        #right since increasing the output will make the cost decrease
        
class Sigmoid:
    def forward(self, inputs):
        self.outputs = 1 / (1+np.exp(-inputs))
        
    def backward(self, dval):
        self.dinputs = dval * (self.outputs * (1 - self.outputs))

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.inputs >= 0)
class dense:
    def __init__(self, input, neuron, regularization_lambda = 0):
        self.weights = 0.01 * np.random.randn(input, neuron)
        self.biases = np.zeros([1, neuron])
        self.regularization_lambda = regularization_lambda
        
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dval):
        self.dweights = np.dot(self.inputs.T, dval)
        self.dbiases = np.sum(dval, axis = 0, keepdims = True)
        self.dinputs = np.dot(dval, self.weights.T)
        #derivative of neuron with respect to input
        #is weight. This makes sense because
        #w is the slope of the y = wX + b
        #and derivative is slope.
        if self.regularization_lambda > 0:
            self.dweights += 2 * self.regularization_lambda * self.weights
            self.dbiases += 2 * self.regularization_lambda * self.biases
        #penalization of neural network due to overfitting
        #this is the backward function so it's 2 * parameter * lambda
        #derivatives are a given.
        #This will increase the cost thus penalizes the model
    
    def save_params(self, filename):
        np.savez(filename, weights = self.weights, biases = self.biases)
        
    def load_params(self, filename):
        data = np.load(filename)
        self.weights = data["weights"]
        self.biases = data["biases"]

class dropout:
    def __init__(self, probability):
        #prob of keeping neuron
        self.probability = probability
        
    def forward(self, inputs, training = True):
        if training:
            self.mask = np.random.binomial(1, self.probability, size=inputs.shape)
            #basically 1 trial, probability of success. If it's saved, then
            #the output is 1 and if not, it's 0.
            #size is input to easily multiply the input and the binomial.
            #if it's not saved, the output is 0 which means anything times 0
            #is 0
            self.outputs = (inputs * self.mask) / self.probability
            #divide by the probability because we have the scale the outputs up
            #since if we remove some neurons, then the output will be lower
            #cuz there's less w * x + w * x
            #so if the probability is 50% to drop, then we would also predict that the neuron
            #in next layer result would also be halved since there's 50% less values.
            #so we basically double the next layer's output which makes it normal
        else:
            self.outputs = inputs.copy()
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        #dropped neurons have gradient 0 so they don't contribute
        #that's why we keep mask
        
class costMAE:
    def forward(self, prediction, real):
        return np.mean(np.abs(prediction - real))
    
    def backward(self, prediction, real):
        self.dinputs = np.sign(prediction - real) / prediction.size
    
class costMSE:
    def forward(self, prediction, real):
        return np.mean(np.square(prediction - real))
    
    def backward(self, prediction, real):
        self.dinputs = (2 / prediction.size) * (prediction - real)

class costBCE:
    def forward(self, prediction, real):
        prediction_clipped = np.clip(prediction, 1e-7, 1-1e-7)
        return np.mean((real * -np.log(prediction_clipped)) + ((1 - real) * -np.log(1 - prediction_clipped)))
    
    def backward(self, prediction, real):
        prediction_clipped = np.clip(prediction, 1e-7, 1-1e-7)
        self.dinputs = -(real / prediction_clipped) + ((1 - real) / (1 - prediction_clipped))
        self.dinputs /= len(prediction_clipped)

class Categorical_Cross_Entropy_Cost:
    def forward(self, pred, real):
        pred_clipped = np.clip (pred, 1e-7, 1)
        correctClass = pred_clipped[range(len(real)), real]
        #basically it'll index through the rows until the end
        #(the end is shown by range(len(real)))
        #and it'll index through those rows while being in the "real" column
        #in other words, real is a number which is the correct class
        #so it'll go through and see the probabilities
        return np.mean(-np.log(correctClass))
        #reason for this calculation is from BCE
        #the costs are literally just -ln(probability)
    
class SGD_Optimizer:
    def __init__(self, learning_rate, mu = 0):
        self.learnRate = learning_rate
        self.mu = mu
    
    def update_params(self, layer):
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_bias = np.zeros_like(layer.biases)
            
        layer.v_weights = (self.mu * layer.v_weights) + (self.learnRate * -layer.dweights)
        layer.v_bias = (self.mu * layer.v_bias) + (self.learnRate * -layer.dbiases)
        layer.weights += layer.v_weights
        layer.biases += layer.v_bias

class Learning_Rate_Decayer:
    def __init__(self, optimizer, decay_factor):
        self.epochs = 0
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.decay_factor = decay_factor
    
    def update_learning_rate(self):
        self.epochs += 1
        self.optimizer.lr = self.initial_lr / (1 + (self.decay_factor * self.epochs))

class Adam_Optimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):    # rho is decay rate
        self.epochs = 0
        self.lr = learning_rate
        self.beta1 = beta1    # beta1 is coefficient of friction for momentum
        self.beta2 = beta2    # beta2 is decay rate for RMSProp
        self.eps = eps     # epsilon
    
    def update_params(self, layer):    # dense layer
        self.epochs += 1
        # if layer does not have the attribute "v_weights", the layer also does not have
        # the attributes "v_biases", "cache_weights", and "cache_biases"
        # we will give the let's initialize those attributes with cache as 0
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        
        # velocities
        layer.v_weights = (layer.v_weights * self.beta1) + ((1 - self.beta1) * layer.dweights * 2)
        layer.v_biases = (layer.v_biases * self.beta1) + ((1 - self.beta1) * layer.dbiases)

        # velocity corrections
        layer.v_weights_corrected = layer.v_weights / (1 - (self.beta1 ** self.epochs))
        layer.v_biases_corrected = layer.v_biases / (1 - (self.beta1 ** self.epochs))

        # caches
        layer.cache_weights = (layer.cache_weights * self.beta2) + ((1 - self.beta2) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.beta2) + ((1 - self.beta2) * layer.dbiases ** 2)

        # cache corrections
        layer.cache_weights_corrected = layer.cache_weights / (1 - (self.beta2 ** self.epochs))
        layer.cache_biases_corrected = layer.cache_biases / (1 - (self.beta2 ** self.epochs))

        # update
        layer.weights += (self.lr / (np.sqrt(layer.cache_weights_corrected) + self.eps)) * -layer.v_weights_corrected
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases_corrected) + self.eps)) * -layer.v_biases_corrected

class RMSProp_Optimizer:
    def __init__(self, learning_rate, rho = 0.9, eps = 1e-7):
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        
    def update_params(self, layer):
        if hasattr(layer, "cache_weights") == False:
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        layer.cache_weights = (layer.cache_weights * self.rho) + ((1 - self.rho) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.rho) + ((1 - self.rho) * layer.dbiases ** 2)
        
        layer.weights += (self.lr / (np.sqrt(layer.cache_weights) + self.eps)) * -layer.dweights
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases) + self.eps)) * -layer.dbiases

def standardize(x, mean = None, std = None):
    if mean is None and std is None:
        mean, std = np.mean(x, axis = 0), np.std(x, axis = 0)
    standardized = (x-mean) / std
    return standardized, (mean, std)