import math

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
