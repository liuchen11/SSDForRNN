import math
import randSVD

import numpy as np

def sharp(input):
	U,s,V=np.linalg.svd(input)
	V=V[:,0:U.shape[1]].transpose()
	return np.dot(U,V)*np.sum(s)