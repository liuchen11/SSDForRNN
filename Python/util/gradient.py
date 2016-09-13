import math
import random

import numpy as np

max_dsigmoid=0.25
max_drelu=1

def sigmoid(M):
    def sigmoid_unit(v):
        return 1.0/(1.0+math.exp(-v))
    sigmoid_vec=np.vectorize(sigmoid_unit)
    return sigmoid_vec(M)

def dsigmoid(M):
    def dsigmoid_unit(v):
        sig_v=1.0/(1.0+math.exp(-v))
        return sig_v*(1-sig_v)
    dsigmoid_vec=np.vectorize(dsigmoid_unit)
    return dsigmoid_vec(M)

def relu(M):
    def relu_unit(v):
        relu_v=v if v>0 else 0
        return relu_v
    relu_vec=np.vectorize(relu_unit)
    return relu_vec(M)

def drelu(M):
    def drelu_unit(v):
        if v<-1e-8:
            return 0
        elif v>1e-8:
            return 1
        else:
            return random.random()
    drelu_vec=np.vectorize(drelu_unit)
    return drelu_vec(M)
