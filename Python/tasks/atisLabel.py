import sys
sys.path.insert(0,'./models/')
sys.path.insert(0,'./util/')
sys.path.insert(0,'./methods/')

import RNN
import sgd_const_lr
import ssd_const_lr
import loader
import time
import numpy as np

if len(sys.argv)!=8:
	print 'Usage: python atisLabel.py <mode> <U_lr> <W_lr> <V_lr> <s_lr> <a/w> <output_file>'
	exit(0)

param={}
param['mode']=sys.argv[1]
param['trainXFile']='../atis/train_word1000.csv'
param['trainYFile']='../atis/train_label1000.csv'
param['dictFile']='../atis/dict10.csv'
param['vectorDim']=10
param['window']=3
param['inputs']=param['vectorDim']*param['window']
param['hiddens']=50
param['outputs']=128
param['batch']=25
param['nEpoch']=10
param['learnRates']={'U':float(sys.argv[2]),'W':float(sys.argv[3]),
	'V':float(sys.argv[4]),'s':float(sys.argv[5])}
param['leftPad']=param['window']/2
param['outputMode']=sys.argv[6]
param['outfile']=sys.argv[7]

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

results='''mode=%s,U_lr=%.4f,W_lr=%.4f,V_lr=%.4f,s_lr=%.4f\n'''%(param['mode'],param['learnRates']['U'],
	param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'])
print results,

if param['mode']=='sgd_const_lr':
	begin=time.time()
	for epoch in xrange(param['nEpoch']):
		err_total=0.0
		for index in xrange(train_num):
			if len(train_index[index])>0:
				err=sgd_const_lr.sgd(rnn,trainX[index],trainY[index])
				err_total+=err
			if index%param['batch']==0:
				rnn.update(param['learnRates'])
				# print 'epoch=%d,index=%d/%d'%(epoch,index,train_num)
				# print 'accumulated error=%f'%err_total
		rnn.update(param['learnRates'])
		print 'epoch %d completed!'%epoch
		print 'total error=%f'%err_total
		results+='|%f'%err_total
	results+='\n'
	end=time.time()
	results+='time=%.2f\n'%(end-begin)

if param['mode']=='ssd_const_lr':
	begin=time.time()
	for epoch in xrange(param['nEpoch']):
		err_total=0.0
		for index in xrange(train_num):
			if len(train_index[index])>0:
				err=ssd_const_lr.ssd(rnn,trainX[index],trainY[index])
				err_total+=err
			if index%param['batch']==0:
				rnn.update(param['learnRates'])
		rnn.update(param['learnRates'])
		print 'epoch %d completed!'%epoch
		print 'total error=%f'%err_total
		results+='|%f'%err_total
	results+='\n'
	end=time.time()
	results+='time=%.2f\n'%(end-begin)

with open(param['outfile'],param['outputMode']) as fopen:
	fopen.write(results)

