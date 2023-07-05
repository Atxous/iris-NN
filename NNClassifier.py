import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ai_Classes as ai
np.random.seed(0)

dataset = pd.read_csv("iris.csv")
classes = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
dataset["class"] = dataset["class"].replace(classes)
dataset = dataset.to_numpy(dtype = float)

x, y = dataset[:,:-1], dataset[:, -1].astype(int)

class Neural_Network:
    def __init__(self, inputs, hidden, outputs):
        self.hidden_layer = ai.dense(inputs, hidden)
        self.activation_layer = ai.ReLU()
        self.hidden_layer2 = ai.dense(hidden, hidden)
        self.activation_layer2 = ai.ReLU()
        self.output_layer = ai.dense(hidden, outputs)
        self.combo = ai.Softmax_Cross_Entropy()
        
        self.trainable_layers = [self.hidden_layer, self.hidden_layer2, self.output_layer]
        
    def forward(self, inputs, true):
        self.hidden_layer.forward(inputs)
        self.activation_layer.forward(self.hidden_layer.outputs)
        self.hidden_layer2.forward(self.activation_layer.outputs)
        self.activation_layer2.forward(self.hidden_layer2.outputs)
        self.output_layer.forward(self.activation_layer2.outputs)
        self.cost = self.combo.forward(self.output_layer.outputs, true)
        
    def backward(self, true):
        self.combo.backward(true)
        self.output_layer.backward(self.combo.dinputs)
        self.activation_layer2.backward(self.output_layer.dinputs)
        self.hidden_layer2.backward(self.activation_layer2.dinputs)
        self.activation_layer.backward(self.hidden_layer2.dinputs)
        self.hidden_layer.backward(self.activation_layer.dinputs)

def get_acc(pred, true):
    predicted = np.argmax(pred, axis = 1)
    #finding, for each row, which has the highest probability (index)
    #then we're comparing that to whether or not it's right
    #result is a column of answers
    return np.mean(predicted == true) * 100
    #matrix that compares whether it's right or not

model = Neural_Network(4, 16, 3)
layers = ["hidden1", "hidden2", "output"]
for layer, name in zip(model.trainable_layers, layers):
    layer.load_params("saved/" + name + ".npz")

data = np.load("saved/standardization.npz")
mean, std = data["mean"], data["std"]
x, _ = ai.standardize(x, mean = mean, std = std)

#model's acc

model.forward(x, y)
print(f"Accuracy: {get_acc(model.output_layer.outputs, y)}%")