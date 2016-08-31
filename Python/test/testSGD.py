import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
sys.path.insert(0,'methods/')

import RNN
import sgd_const_lr as sgd
import random
import time
import numpy as np

N=10
H=5
K=10
S=50
iters=1000

rnn=RNN.RNN(N,H,K)

states=np.random.random([S,N])*2-1
ground_truth=np.zeros([S,K])

for i in xrange(S):
    spot=random.randint(0,K-1)
    ground_truth[i,spot]=1

duration=0
for i in xrange(iters):
    begin=time.time()
    err=sgd.sgd(rnn,states,ground_truth)
    end=time.time()
    rnn.update({'W':0.5,'U':0.5,'V':0.5,'s':0.5})
    duration+=end-begin
    print i,'\t',err
print 'time: ', duration
