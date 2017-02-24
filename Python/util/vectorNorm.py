import math
import numpy as np


def norm(input,n):
    # vectorize
    inp=np.array(input).reshape(-1)

    if n==np.inf:
        return np.max(inp)
    elif n==2:
        p=np.dot(inp,inp)
        return math.sqrt(p)
    return np.linalg.norm(inp,n)