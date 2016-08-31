import sys
sys.path.insert(0,'util/')

import softmax
import numpy as np

vector1=np.ones(100)
vectors=np.random.random([100,100])

print softmax.softmax(vector1)
for i in xrange(100):
    vector=softmax.softmax(vectors[i])
    assert(np.sum(vector)>0.999 and np.sum(vector)<1.001)
