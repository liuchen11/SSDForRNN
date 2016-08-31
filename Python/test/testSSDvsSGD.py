import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
sys.path.insert(0,'methods/')

import RNN
import sgd_const_lr
import ssd_const_lr
import softmax
import random
import time
import cPickle
import matplotlib.pyplot as plt
import numpy as np

N=10
H=5
K=10
S=50
nEpoch=1000

rnn1=RNN.RNN(N,H,K)
rnn2=rnn1.copy()
sgd_lr={'W':0.1,'U':0.1,'V':0.1,'s':0.2}
ssd_lr={'W':0.1,'U':0.1,'V':0.1,'s':0.2}

states=np.random.randn(S,N)
ground_truth=np.zeros([S,K])

for i in xrange(S):
    spot=random.randint(0,K-1)
    ground_truth[i,spot]=1

data=cPickle.load(open('data/50x10in10out'))
states=data['states']
ground_truth=data['ground_truth']
sgd_errs=[]
ssd_errs=[]


duration1=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err1=sgd_const_lr.sgd(rnn1,states,ground_truth)
    end=time.time()
    rnn1.update(sgd_lr)
    duration1+=end-begin
    print i,'\t',err1
    sgd_errs.append(err1)
print 'time:','\t',duration1

duration2=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err2=ssd_const_lr.ssd(rnn2,states,ground_truth)
    end=time.time()
    rnn2.update(ssd_lr)
    duration2+=end-begin
    print i,'\t',err2
    ssd_errs.append(err2)
print 'time:','\t',duration2

plt.plot(range(nEpoch),sgd_errs,'b',range(nEpoch),ssd_errs,'r')
plt.show()
