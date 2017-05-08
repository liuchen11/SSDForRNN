import sys
import copy
import time
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
sys.path.insert(0,'methods/')

import RNNs
import grads
import matrixNorm
import optimizers

N=10
H=[5,]
K=10
S=50
nEpoch=1000

rnn1=RNNs.RNNs([N,]+H+[K,],['tanh',])
rnn2=rnn1.copy()

sgd_optimizer=optimizers.sgdConstOptimizer(rnn1)
ssd_optimizer=optimizers.ssdConstOptimizer(rnn2)


sgd_lr={'W':.5,'U':.5,'V':.5,'s':.5}
ssd_lr={'W':.2,'U':.2,'V':.5,'s':.5}

data=cPickle.load(open('../toy/50x10in10out'))
states=data['states']
ground_truth=data['ground_truth']
sgd_const_err=[]
ssd_const_err=[]

sgd_const_time=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err=grads.gradient(rnn1,states,ground_truth)
    sgd_optimizer.update(rnn1,copy.copy(sgd_lr))
    end=time.time()
    sgd_const_time+=end-begin
    print i,'\t',err,'\t',matrixNorm.norm(rnn1.W[0],np.inf)#,'\t',matrixNorm.norm(rnn1.U[0],np.inf)
    sgd_const_err.append(err)
print 'time','\t',sgd_const_time

ssd_const_time=0.0
for i in xrange(nEpoch):
    begin=time.time()
    err=grads.gradient(rnn2,states,ground_truth)
    ssd_optimizer.update(rnn2,copy.copy(ssd_lr))
    end=time.time()
    ssd_const_time+=end-begin
    print i,'\t',err,'\t',matrixNorm.norm(rnn2.W[0],np.inf)#,'\t',matrixNorm.norm(rnn2.U[0],np.inf)
    ssd_const_err.append(err)
print 'time','\t',ssd_const_time

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(range(1,len(sgd_const_err)+1),sgd_const_err,'blue',label='SGD_const')
ax.plot(range(1,len(ssd_const_err)+1),ssd_const_err,'red',label='SSD_const')

handles,labels=ax.get_legend_handles_labels()
by_label=OrderedDict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys())
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()

