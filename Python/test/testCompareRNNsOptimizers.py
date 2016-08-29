import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
sys.path.insert(0,'methods/')
import cPickle
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import RNNs
import grads
import optimizers

N=10
H=[5,]
K=10
S=50
nEpoch=3000

rnn1=RNNs.RNNs([N,]+H+[K,])
rnn2=rnn1.copy()

sgd_optimizer=optimizers.constOptimizer(rnn1)
ssd_optimizer=optimizers.constOptimizer(rnn2)

sgd_lr={'W':0.2,'U':0.2,'V':0.2,'s':0.1}
ssd_lr={'W':0.2,'U':0.2,'V':0.2,'s':0.1}

data=cPickle.load(open('../data/50x10in10out'))
states=data['states']
ground_truth=data['ground_truth']
sgd_const_err=[]
ssd_const_err=[]

sgd_const_time=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err=grads.sgd(rnn1,states,ground_truth,sgd_optimizer)
    sgd_optimizer.update(rnn1,sgd_lr)
    end=time.time()
    sgd_const_time+=end-begin
    print i,'\t',err
    sgd_const_err.append(err)
print 'time','\t',sgd_const_time

ssd_const_time=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err=grads.ssd(rnn2,states,ground_truth,ssd_optimizer)
    ssd_optimizer.update(rnn2,ssd_lr)
    end=time.time()
    ssd_const_time+=end-begin
    print i,'\t',err
    ssd_const_err.append(err)
print 'time','\t',ssd_const_time

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(range(1,nEpoch+1),sgd_const_err,'blue',label='SGD_const')
ax.plot(range(1,nEpoch+1),ssd_const_err,'red',label='SSD_const')

handles,labels=ax.get_legend_handles_labels()
by_label=OrderedDict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys())
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()