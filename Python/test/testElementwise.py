import sys
sys.path.insert(0,'util/')

from elementwise import *

import numpy as np
import time

instances=50
matrix=np.random.random([instances,1000,1000])
divisor=np.random.random([instances,1000,1000])
result1=np.zeros([instances,1000,1000])
result2=np.zeros([instances,1000,1000])

'>>> Test Power'
begin=time.time()
for i in xrange(instances):
    result1[i]=elmtwiseprod(matrix[i],matrix[i])
end=time.time()
print 'used %.2f seconds'%(end-begin)

begin=time.time()
for i in xrange(instances):
    result2[i]=np.power(matrix[i],2)
end=time.time()
print 'used %.2f seconds'%(end-begin)

'''check accuracy'''
for i in xrange(instances):
    diff=np.linalg.norm(result2[i]-result1[i])
    assert(diff<1e-5)

'>>> Test sqrt'
begin=time.time()
for i in xrange(instances):
    result1[i]=elmtwisesqrt(matrix[i])
end=time.time()
print 'used %.2f seconds'%(end-begin)

begin=time.time()
for i in xrange(instances):
    result2[i]=np.sqrt(matrix[i])
end=time.time()
print 'used %.2f seconds'%(end-begin)

'''check accuracy'''
for i in xrange(instances):
    diff=np.linalg.norm(result2[i]-result1[i])
    assert(diff<1e-5)

'>>>Test Division'
begin=time.time()
for i in xrange(instances):
    result1[i]=elmtwisediv(matrix[i],divisor[i])
end=time.time()
print 'used %.2f seconds'%(end-begin)

begin=time.time()
for i in xrange(instances):
    result2[i]=np.divide(matrix[i],divisor[i])
end=time.time()
print 'used %.2f seconds'%(end-begin)

'''check accuracy'''
for i in xrange(instances):
    diff=np.linalg.norm(result2[i]-result1[i])
    assert(diff<1e-5)