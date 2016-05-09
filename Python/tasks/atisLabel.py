import sys
sys.path.insert(0,'./models/')
sys.path.insert(0,'./util/')
sys.path.insert(0,'./optimization')

import RNN
import sgd
import loader
import numpy as np

param={}
param['trainXFile']='../atis/train_word1000.csv'
param['trainYFile']='../atis/train_label1000.csv'
param['dictFile']='../atis/dict10.csv'
param['vectorDim']=10
param['window']=3
param['inputs']=param['vectorDim']*param['window']
param['hiddens']=10
param['outputs']=128
param['batch']=25
param['nEpoch']=5
param['sgdLearnRate1']=1
param['sgdLearnRate2']=1
param['leftPad']=param['window']/2

train_index=loader.loadData(param['trainXFile'])
train_label=loader.loadData(param['trainYFile'])
dictionary=loader.loadDict(param['dictFile'])
dictionary[-1]=np.random.randn(param['vectorDim'])*0.5

rnn=RNN.RNN(param['inputs'],param['hiddens'],param['outputs'])

train_num=len(train_index)
trainX=[]
trainY=[]

for i in xrange(train_num):
	sentence_len=len(train_index[i])
	single_input=np.zeros([sentence_len,param['inputs']])
	single_label=np.zeros([sentence_len,param['outputs']])

	for p in xrange(sentence_len+param['window']-1):
		toFill=dictionary[-1]
		if p>=param['leftPad'] and p<sentence_len+param['leftPad']:
			word=train_index[i][p-param['leftPad']]
			if dictionary.has_key(word)==False:
				dictionary[word]=np.random.randn(param['vectorDim'])*0.05
			toFill=dictionary[word]
		startline=max(0,p-param['window']+1)
		endline=min(sentence_len-1,p)
		for line in range(startline,endline+1):
			col=p-line
			single_input[line,col*param['vectorDim']:(col+1)*param['vectorDim']]=toFill

	for w in xrange(sentence_len):
		single_label[w,train_label[i][w]]=1
	trainX.append(single_input)
	trainY.append(single_label)

for epoch in xrange(param['nEpoch']):
	err_total=0.0
	for index in xrange(train_num):
		if len(train_index[index])>0:
			err=sgd.sgd(rnn,trainX[index],trainY[index])
			err_total+=err
		if index%param['batch']==0:
			rnn.update(param['sgdLearnRate1'],param['sgdLearnRate2'])
			# print 'epoch=%d,index=%d/%d'%(epoch,index,train_num)
			# print 'accumulated error=%f'%err_total
	rnn.update(param['sgdLearnRate1'],param['sgdLearnRate2'])
	print 'epoch %d completed!'%epoch
	print 'total error=%f'%err_total