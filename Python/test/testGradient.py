import sys
sys.path.insert(0,'./util/')
import numpy as np

import gradient

M=np.zeros([3,3])
print(gradient.sigmoid(M))
print(gradient.dsigmoid(M))
print(gradient.ddsigmoid(M))

