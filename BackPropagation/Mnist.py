# -*- coding: utf-8 -*-
"""
Created on Sat 

@author: gaurav
"""

from sklearn.datasets import load_digits
import numpy as np
import nn
import optim2
import loss
from autodiff import *
from matplotlib import pyplot as plt


def one_hot(n, max):
    arr = [0] * max
    
    arr[n] = 1
    
    return arr

mnist = load_digits()
images = np.array([image.flatten() for image in mmnist.images])
targets = np.array([one_hot(n , 10) for n in mnist.target])

 #one_hot(3, 10) => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#Building the model
model = nn.Model([
    nn.Linear(64),
    nn.Tanh(),
    nn.Linear(32),
    nn.Sigmoid(),
    nn.Linear(10),
    nn.Softmax()
    ])


##Training the model
model.train(images[:1000], targets[:1000], epochs=50, loss_fn = loss.CategoricalCrossentropy, optimizer = optim.RMSProp(0.001),batch_size=128)


## Testing the model
images = images[1000:]
np.random.shuffle(images)

correct = 0
cmt = 0

for image in images:
    pred = np.argmax(model(np.array([image])).value, axis=1)
    real = np.argmax(target[cnt])
    if pred[0] == real:
        correct += 1
    cmt += 1
    
print("accuracy:", correct / cnt)
    