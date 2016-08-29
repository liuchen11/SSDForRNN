import sys
sys.path.insert(0,'util/')
import numpy as np

import vectorNorm
import time

vectors=np.random.random([10000,10000])
results=np.zeros(len(vectors))
begin=time.time()
for i in xrange(len(vectors)):
	results[i]=vectorNorm.norm(vectors[i],np.inf)
end=time.time()
print 'Completed in %.2f seconds'%(end-begin)
