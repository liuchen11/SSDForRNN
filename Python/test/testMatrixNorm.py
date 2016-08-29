import sys
sys.path.insert(0,'util/')

import matrixNorm
import numpy as np

M=np.random.random([2000,2000])
print(matrixNorm.norm(M,1))
