import math

import numpy as np

def sharp(input):
    U,s,V=np.linalg.svd(input,full_matrices=False)
    return np.dot(U,V)*np.sum(s)

