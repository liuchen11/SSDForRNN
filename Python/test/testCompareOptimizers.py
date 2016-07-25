import sys
sys.path.insert(0,'../util/')
sys.path.insert(0,'../models/')
sys.path.insert(0,'../methods/')
import cPickle
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import RNN
import grad
import optimizer

N=10
H=5
K=10
S=50
nEpoch=20000

rnn1=RNN.RNN(N,H,K)
rnn2=rnn1.copy()
rnn3=rnn1.copy()
rnn4=rnn1.copy()
rnn5=rnn1.copy()
rnn6=rnn1.copy()
sgd_lr={'W':0.05,'U':0.05,'V':0.05,'s':0.1}
ssd_lr={'W':0.05,'U':0.05,'V':0.05,'s':0.1}

data=cPickle.load(open('../data/50x10in10out'))
states=data['states']
ground_truth=data['ground_truth']
sgd_const_err=[]
ssd_const_err=[]
sgd_adagrad_err=[]
ssd_adagrad_err=[]
sgd_rms_err=[]
ssd_rms_err=[]

sgd_const_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn1,states,ground_truth)
	optimizer.sgd_const(rnn1,sgd_lr)
	end=time.time()
	sgd_const_time+=end-begin
	print i,'\t',err
	sgd_const_err.append(err)
print 'time:','\t',sgd_const_time

ssd_const_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn2,states,ground_truth)
	optimizer.ssd_const(rnn2,ssd_lr)
	end=time.time()
	ssd_const_time+=end-begin
	print i,'\t',err
	ssd_const_err.append(err)
print 'time:','\t',ssd_const_time

sgd_adagrad_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn3,states,ground_truth)
	optimizer.sgd_adagrad(rnn3,sgd_lr)
	end=time.time()
	sgd_adagrad_time+=end-begin
	print i,'\t',err
	sgd_adagrad_err.append(err)
print 'time:','\t',sgd_adagrad_time

ssd_adagrad_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn4,states,ground_truth)
	optimizer.ssd_adagrad(rnn4,ssd_lr)
	end=time.time()
	ssd_adagrad_time+=end-begin
	print i,'\t',err
	ssd_adagrad_err.append(err)
print 'time:','\t',ssd_adagrad_time

sgd_rms_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn5,states,ground_truth)
	optimizer.sgd_rms(rnn5,sgd_lr)
	end=time.time()
	sgd_rms_time+=end-begin
	print i,'\t',err
	sgd_rms_err.append(err)
print 'time:','\t',sgd_rms_time

ssd_rms_time=0.0
for i in xrange(nEpoch):
	begin=time.time()
	err=grad.grad(rnn6,states,ground_truth)
	optimizer.ssd_rms(rnn6,ssd_lr)
	end=time.time()
	ssd_rms_time+=end-begin
	print i,'\t',err
	ssd_rms_err.append(err)
print 'time:','\t',ssd_rms_time

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(range(1,nEpoch+1),sgd_const_err,'blue',label='SGD_Const')
ax.plot(range(1,nEpoch+1),ssd_const_err,'red',label='SSD_Const')
ax.plot(range(1,nEpoch+1),sgd_adagrad_err,'black',label='SGD_ADA')
ax.plot(range(1,nEpoch+1),ssd_adagrad_err,'green',label='SSD_ADA')
ax.plot(range(1,nEpoch+1),sgd_rms_err,'yellow',label='SGD_RMS')
ax.plot(range(1,nEpoch+1),ssd_rms_err,'cyan',label='SSD_RMS')

handles,labels=ax.get_legend_handles_labels()
by_label=OrderedDict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys())
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()
