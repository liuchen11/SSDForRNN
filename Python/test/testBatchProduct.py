import sys
sys.path.insert(0,'util/')
import batchProduct

import numpy as np

m1=np.random.random([3,3,4]).astype(np.float32)
m2=np.random.random([4,3]).astype(np.float32)

print(m1)
print(m2)

m12=batchProduct.nXone(m1,m2)
m21=batchProduct.oneXn(m2,m1)
print(m12)
print(m21)

print np.dot(m1[1],m2)
