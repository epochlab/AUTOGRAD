#!/usr/bin/env python3

import math
import numpy as np

class Value:
    """ Stores a single scalar value and its gradient """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now."
        out = Value(self.data**other, (self,), f'**{other}', '^')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def matmul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out

    def softmax(self):
        out =  Value(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data

        def _backward():
            self.grad += (out.grad - np.reshape(np.sum(out.grad * softmax, 1), [-1, 1])) * softmax
        out._backward = _backward

        return out

    def log(self):
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward

        return out

    def reduce_sum(self, axis=None):
        out = Value(np.sum(self.data, axis=axis), (self,), 'reduce_sum')
        
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)        
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __repr__(self):
        return f"Value(data={self.data})"

    def shape(self):
        return self.data.shape