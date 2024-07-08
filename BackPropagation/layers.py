# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:50:10 2024

@author: gaurav
"""

import numpy as np
import loss
import optim
import autodiff as ad
import layers
from tqdm import tqdm

class Activation:
    def __init__(self):
        pass
    
class Layer:
    def __init__(self):
        pass
    
class Model:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
            
        return output
    
    def train(self, x, y, epochs=10, loss_fn = loss.MSE, optimizer=optim.SGD(lr=0.1), batch_size=32):
        for epoch in range(epochs):
            _loss = 0
            print ("**")
            print (" ")
            print (f"EPOCH", epoch + 1)
            for batch in tqdm(range(0, len(x), batch_size)):
                output = self(x[batch:batch+batch_size])
                l = loss_fn(output, y[batch:batch+batch_size])
                optimizer(self, l)
                _loss += l
            
            print ("LOSS", _loss.value)
            print (" ")
    
class Linear(Layer):
    def __init__(self, units):
        self.units = units
        self.initialized = False
        
    def __call__(self, x):
        self.input = x
        if not self.initialized:
            self.w = np.random.rand(self.input.shape[-1], self.units)
            self.b = np.random.rand(self.units)
            self.initialized = True
            
        return self.input @ self.w + self.b

    def backward(self, grad):
        self.w_gradient = self.input.T @ grad
        self.b_gradient = np.sum(grad, axis=0)
        return grad @ self.w.T 
        
class Sigmoid(Activation):
    def __call__(self, x):
        self.output = 1/ (1 + np.exp(-x))
        
        return self.output
    
    def backward(self, grad):
        return grad * (self.output * (1 - self.output))
    
class Relu(Activation):
    def __call__(self,x):
        self.output = np.maximum(0,x)
        return self.output
    
    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)
    
class Softmax(Activation):
    def __call__(self, x):
        exps = np.exp(x - np.max(x))
        self.output = exps / np.sum(exps, axis =1, keepdims=True)
        return self.output
    
    def backward(self, grad):
        m,n = self.output.shape
        p = self.output
        tensor1 = np.einsum('ij, ik->ijk',  p, p)
        tensor2 = np.einsum('ij, jk-> ijk', p, np.eye(n,n))
        dSoftmax  = tensor2 - tensor1
        dz = np.einsum('ijk, ik->ij', dSoftmax, grad)
        return dz
    
class Tanh(Activation):
    def __call__(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad):
        return grad * (1- self.output ** 2)