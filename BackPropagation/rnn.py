# -*- coding: utf-8 -*-
"""
Created on

@author: gaurav
"""

import numpy as np
import nn
import optim2
import optim
import loss
from matplotlib import pyplot as plt

x_axis = np.array(list(range(200))).T
seq = [np.sin(i) for i in range(200)]


x = []
y = []

for i in range(len(seq) - 50):
    new_seq = [[i] for i in seq[i:i+50]]
    x.append(new_seq)
    y.append([seq[i + 50]])
    
## Building model
model = nn.Model([
    nn.RNN(32, hidden_dim=32, return_sequences= True),
    nn.RNN(0, hidden_dim=32),
    nn.Linear(8),
    nn.Linear(1)
    ])


model.train(np.array(x[:50]), np.array(y[:50]), epochs=10, optimizer = optim.RMSProp(0.0005), loss_fn=loss.MSE, batch_size=8)

preds = []

for i in x:
    preds.append(model(np.expand_dims(np.array(i), axis=0)).value[0][0])
    
plt.plot(x_axis[:150], seq[:150])
plt.plot(x_axis[:150], preds)
plt.savefig("graph.png")

