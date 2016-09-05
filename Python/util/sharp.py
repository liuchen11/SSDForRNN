import math
import randSVD

import numpy as np

def sharp(input):
    U,s,V=np.linalg.svd(input,full_matrices=False)
    # U=U[:,0:V.shape[1]]
    # V=V[0:U.shape[0],:]
    return np.dot(U,V)*np.sum(s)

