# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:47:06 2024

@author: gaurav
"""
import autodiff as ad
import numpy as np

class MSE:
    def __call__(self, pred, y):
        self.error = pred - y
        return np.mean(self.error ** 2)
    
    def CategoricalCrossentropy(pred, real):
        loss = -1 * ad.reduce_mean(real * ad.log(pred))
        
        return loss