# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:16:51 2024

@author: gaurav
"""
import layers
import loss
import optim
import numpy as np


x = np.array([[0,1], [0,0], [1,1], [1,0]])
y = np.array([[0,1], [1,0], [1,0], [0,1]])

net = layers.Model([
    layers.Linear(8),
    
    layers.Linear(4),
    layers.Sigmoid(),
    layers.Linear(2),
    layers.Softmax()
    ])

net.train(x,y, optim=optim.RMSProp(lr=0.6), loss=loss.MSE(), epochs= 400)

print (net(x))
