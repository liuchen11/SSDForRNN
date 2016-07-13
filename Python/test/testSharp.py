import sys
sys.path.insert(0,'../util/')

import sharp
import matrixNorm
import numpy as np

def func(M1,M2):
	dot_product=np.dot(M1.reshape(-1),M2.reshape(-1))
	normalize=matrixNorm.norm(M2,np.inf)**2/2
	return dot_product-normalize

M=np.random.randn(100,200)
M_sharp=sharp.sharp(M)
minV=func(M,M_sharp)

for i in xrange(100):
	rand=np.random.randn(100,200)
	V=func(M,rand)
	assert(V>=minV)
