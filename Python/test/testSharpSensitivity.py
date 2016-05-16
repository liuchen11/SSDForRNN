import sys
sys.path.insert(0,'./util/')

import sharp
import matrixNorm
import random
import numpy as np

'''
Test the sharp operator's sensitivity of rank deficient matrix
'''

column=np.random.randn(100,1)
row=np.random.randn(1,100)
m1=np.dot(column,row)
n1=np.random.randn(100,100)
m2=np.copy(m1)
n2=np.copy(n1)

delta=1e-5
x=random.randint(0,99)
y=random.randint(0,99)
m2[x,y]+=delta
n2[x,y]+=delta

m1_sharp=sharp.sharp(m1)
m2_sharp=sharp.sharp(m2)
n1_sharp=sharp.sharp(n1)
n2_sharp=sharp.sharp(n2)
print np.mean(np.abs(m1_sharp-m2_sharp))
print np.mean(np.abs(n1_sharp-n2_sharp))