import sys
sys.path.insert(0,'./util/')

import matrixNorm
import numpy as np

M=np.random.random([10000,10000])
print(matrixNorm.norm(M,1))