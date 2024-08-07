# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:52:52 2024

@author: gaurav

"""

import numpy as np
import string
import random

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


np.seterr(invalid='ignore')

def is_matrix(o):
    return type(o) == np.ndarray

def same_shape(s1, s2):
    for a, b in zip(s1, s2):
        if a != b:
            return False

    return True

class Tensor:
    __array_priority__ = 1000
    def __init__(self, value, trainable=True):
        self.value = value
        self.dependencies = []
        self.grads = []
        self.grad_value = None
        self.shape = 0
        self.matmul_product = False
        self.gradient = 0
        self.trainable = trainable
        self.id = id_generator()

        if is_matrix(value):
            self.shape = value.shape

    def depends_on(self, target):
        if self == target:
            return True

        dependencies = self.dependencies

        for dependency in dependencies:
            if dependency == target:
                return True
            elif dependency.depends_on(target):
                return True

        return False

    def __mul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value * other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value)
        var.grads.append(self.value)
        return var

    def __rmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value * other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value)
        var.grads.append(self.value)
        return var

    def __add__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value + other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(np.ones_like(other.value))
        return var

    def __radd__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value + other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(np.ones_like(other.value))
        return var

    def __sub__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other)

        var = Tensor(self.value - other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(-np.ones_like(other.value))
        return var

    def __rsub__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(other.value - self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)
        var.grads.append(np.ones_like(other.value))
        var.grads.append(-np.one_like(self.value))
        return var

    def __pow__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value ** other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)

        grad_wrt_self = other.value * self.value ** (other.value - 1)
        var.grads.append(grad_wrt_self)

        grad_wrt_other = (self.value ** other.value) * np.log(self.value)
        var.grads.append(grad_wrt_other)

        return var

    def __rpow__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(other.value ** self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)

        grad_wrt_other = self.value * other.value ** (self.value - 1)
        var.grads.append(grad_wrt_other)

        grad_wrt_self = (other.value ** self.value) * np.log(other.value)
        var.grads.append(grad_wrt_self)

        return var


    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __matmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value @ other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value.T)
        var.grads.append(self.value.T)

        var.matmul_product = True
        return var

    def __rmatmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(other.value @ self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)
        var.grads.append(self.value.T)
        var.grads.append(other.value.T)

        var.matmul_product = True

        return var

    def grad(self, target, grad = None):
        grad = self.value / self.value if grad is None else grad
        grad = np.float32(grad)

        if not self.depends_on(target):
            return 0

        if self == target:
            return grad

        final_grad = 0

        for dependency, _grad in zip(self.dependencies, self.grads):
            local_grad = np.float32(_grad) if dependency.depends_on(target) else 0

            if local_grad != 0:
                if self.matmul_product:                
                    if dependency == self.dependencies[0]:
                        local_grad = grad @ local_grad
                    else:
                        local_grad = local_grad @ grad
                else:
                    if dependency.shape != 0 and not same_shape(grad.shape, local_grad.shape):
                        ndims_added = grad.ndim - local_grad.ndim
                        for _ in range(ndims_added):
                            grad = grad.sum(axis=0)

                        for i, dim in enumerate(local_grad.shape):
                            if dim == 1:
                                grad = grad.sum(axis=i, keepdims=True)

                    local_grad *= grad

            final_grad += dependency.grad(target, local_grad)

        return final_grad


    def get_gradients(self, grad = None):
        grad = np.ones_like(self.value) if grad is None else grad
        grad = np.float32(grad)

        for dependency, _grad in zip(self.dependencies, self.grads):
            if dependency.trainable:
                local_grad = np.float32(_grad)

                if self.matmul_product:                
                    if dependency == self.dependencies[0]:
                        local_grad = grad @ local_grad
                    else:
                        local_grad = local_grad @ grad
                else:
                    if dependency.shape != 0 and not same_shape(grad.shape, local_grad.shape):
                        ndims_added = grad.ndim - local_grad.ndim
                        for _ in range(ndims_added):
                            grad = grad.sum(axis=0)

                        for i, dim in enumerate(dependency.shape):
                            if dim == 1:
                                grad = grad.sum(axis=i, keepdims=True)

                    local_grad = local_grad * np.nan_to_num(grad)


                dependency.gradient += local_grad
                dependency.get_gradients(local_grad)

    def __repr__(self):
        return f"Tensor ({self.value})"
    
a = Tensor(10)
b = Tensor(5)
c = 2
d = (a*b)**c
d.get_gradients()

print (a.gradient, b.gradient)


def reduce_sum(tensor, axis = None, keepdims=False):
        var = Tensor(np.sum(tensor.value, axis = axis, keepdims=keepdims))
        var.dependencies.append(tensor)
        var.grads.append(np.ones(tensor.value.shape))

        return var

def reduce_mean(tensor, axis = None, keepdims=False):
    return reduce_sum(tensor, axis, keepdims) / tensor.value.size

def log(tensor):
    var = Tensor(np.log(tensor.value))
    var.dependencies.append(tensor)
    var.grads.append(1 / tensor.value)

    return var

def stack(tensor_list):
    tensor_values = [tensor.value for tensor in tensor_list]
    s = np.stack(tensor_values)
    
    var = Tensor(s)
    var.dependencies += tensor_list
    
    for tensor in tensor_list:
        var.grads.append(np.ones(tensor.value.shape))
        
    return var


    