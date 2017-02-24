import sys
sys.path.insert(0,'util/')

import matrixNorm
import numpy as np

M=np.random.random([2000,1000])
print matrixNorm.norm(M,1)
print matrixNorm.norm(M,2)
print matrixNorm.norm(M,np.inf)
