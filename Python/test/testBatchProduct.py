import sys
sys.path.insert(0,'../util/')
import batchProduct

import numpy as np

m1=np.ones([3,3,3])
m2=np.random.random([3,3])

print(m1)
print(m2)

m12=batchProduct.nXone(m1,m2)
m21=batchProduct.oneXn(m2,m1)
print(m12)
print(m21)
