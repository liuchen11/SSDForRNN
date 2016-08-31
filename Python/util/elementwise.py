import math

import numpy as np

def elmtwiseprod(a,b):
    assert(a.shape[0] == b.shape[0])
    elmtwiseprod_vec = np.vectorize(lambda a,b : a*b)
    return elmtwiseprod_vec(a,b)

def elmtwisediv(a,b):
    assert(a.shape[0] == b.shape[0])
    elmtwisediv_vec = np.vectorize(lambda a,b : a/b)
    return elmtwisediv_vec(a,b)

def elmtwisesqrt(a):
    elmtwisesqrt_vec = np.vectorize(lambda a : math.sqrt(a))
    return elmtwisesqrt_vec(a)
