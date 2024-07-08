# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:52:59 2024

@author: gaurav
"""

import layers
import loss
import optim
import numpy as np

x = np.array([[0,1], [0,0], [1,1],[0,1]])
y = np.array([[1],[0],[0],[1]])

net = layers.Model([
    layers.Linear(8),
    layers.Relu(),
    layers.Linear(4),
    layers.Sigmoid(),
    layers.Linear(1),
    layers.Sigmoid()
    ])

net.train(x, y, optim=optim.SGD(lr=0.6), loss=loss.MSE(), epochs=200)

print(net(x))
