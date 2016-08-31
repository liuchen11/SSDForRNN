import math
import numpy as np


def norm(input,n):
    if n==np.inf:
        return np.max(input)
    elif n==2:
        p=np.dot(input,input)
        return math.sqrt(p)
    return np.linalg.norm(input,n)