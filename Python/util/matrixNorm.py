import math
import vectorNorm

import numpy as np

def norm(input,n):
	U,s,V=np.linalg.svd(input)
	return vectorNorm(s,n)