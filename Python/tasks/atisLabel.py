import sys
import os
from random import shuffle,random
sys.path.insert(0,'../models/')
sys.path.insert(0,'../util/')
sys.path.insert(0,'../methods/')

import RNN
import sgd_const_lr
import ssd_const_lr
import rms_const_lr
import loader
import time
import numpy as np

if len(sys.argv)!=8:
	print 'Usage: python atisLabel.py <mode> <U_lr> <W_lr> <V_lr> <s_lr> <a/w> <output_file>'
	exit(0)

maketest=True
decreasingLR=True
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
param['nEpoch']=1000
param['learnRates']={'U':float(sys.argv[2]),'W':float(sys.argv[3]),
	'V':float(sys.argv[4]),'s':float(sys.argv[5])}
param['leftPad']=param['window']/2
param['outputMode']=sys.argv[6]
param['outfile']='train_' + sys.argv[7]
param['test_outfile']='test_' + sys.argv[7]
param['numberOfSave']=9
param['alpha']=random()
param['dcrorcst'] = 'cst'
if decreasingLR:
        param['dcrorcst'] = 'dcr'

train_index=loader.loadData(param['trainXFile'])
train_label=loader.loadData(param['trainYFile'])
dictionary=loader.loadDict(param['dictFile'])
dictionary[-1]=np.random.randn(param['vectorDim'])*0.5

rnn=RNN.RNN(param['inputs'],param['hiddens'],param['outputs'])

if maketest:
        lmbda = random()
        shuffle(train_index, lambda: lmbda)
        shuffle(train_label, lambda: lmbda)

train_num=len(train_index)
trainX=[]
trainY=[]

########################################
#                                      #
#    Debut du formatage des donnees    #
#                                      #
########################################

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

if maketest:
        ttX = trainX
        ttY = trainY
        trainX = ttX[0:900]
        trainY = ttY[0:900]
        testX = ttX[900:1000]
        testY = ttY[900:1000]

        train_num = 900
        test_num = 100
        test_results='''mode=%s,U_lr=%.4f,W_lr=%.4f,V_lr=%.4f,s_lr=%.4f\n'''%(param['mode'],param['learnRates']['U'],
	param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'])

        
results='''mode=%s,U_lr=%.4f,W_lr=%.4f,V_lr=%.4f,s_lr=%.4f\n'''%(param['mode'],param['learnRates']['U'],
	param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'])
print results,

######################
#                    #
#    Debut du SGD    #
#                    #
######################

if param['mode']=='sgd_const_lr':
	begin=time.time()
        epochh = 0
	for epoch in xrange(param['nEpoch']):
                if decreasingLR:
                        epochh = epoch
                if maketest:
                        test_err_total=0.0
		        for index in xrange(test_num):
			        if len(train_index[index+900])>0:
				        test_err=sgd_const_lr.sgd(rnn,testX[index],testY[index],False)
				        test_err_total+=test_err
                        test_err_total*=10
                        test_results+='|%f'%(test_err_total)
                        os.system('''echo '|%s' >> results_saved_%s_test_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(test_err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
                err_total=0.0
		for index in xrange(train_num):
			if len(train_index[index])>0:
				err=sgd_const_lr.sgd(rnn,trainX[index],trainY[index])
				err_total+=err
			if index%param['batch']==0:
				rnn.update(param['learnRates'],epochh+1)
				# print 'epoch=%d,index=%d/%d'%(epoch,index,train_num)
				# print 'accumulated error=%f'%err_total
		rnn.update(param['learnRates'],epochh+1)
		print 'epoch %d completed!'%epoch
                if maketest:
                        print 'total train error=%f'%err_total
                        print 'total test error=%f'%test_err_total
                else:
		        print 'total error=%f'%err_total
		results+='|%f'%err_total
                os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
	results+='\n'
	end=time.time()
	results+='time=%.2f\n'%(end-begin)

        if maketest:
                test_results+='\n'
	        test_results+='time=%.2f\n'%(end-begin)

######################
#                    #
#    Debut du SSD    #
#                    #
######################
        
if param['mode']=='ssd_const_lr':
	begin=time.time()
        epochh = 0
	for epoch in xrange(param['nEpoch']):
                if decreasingLR:
                        epochh = epoch
                if maketest:
                        test_err_total=0.0
		        for index in xrange(test_num):
			        if len(train_index[index+900])>0:
				        test_err=ssd_const_lr.ssd(rnn,testX[index],testY[index],False)
				        test_err_total+=test_err
                        test_err_total*=10
                        test_results+='|%f'%(test_err_total)
                        os.system('''echo '|%s' >> results_saved_%s_test_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(test_err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
		
                err_total=0.0
		for index in xrange(train_num):
			if len(train_index[index])>0:
				err=ssd_const_lr.ssd(rnn,trainX[index],trainY[index])
				err_total+=err
			if index%param['batch']==0:
				rnn.update(param['learnRates'],epochh+1)  
		rnn.update(param['learnRates'],epochh+1)
		print 'epoch %d completed!'%epoch
		if maketest:
                        print 'total train error=%f'%err_total
                        print 'total test error=%f'%test_err_total
                else:
		        print 'total error=%f'%err_total
		results+='|%f'%err_total
                os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
	results+='\n'
	end=time.time()
	results+='time=%.2f\n'%(end-begin)

        if maketest:
                test_results+='\n'
	        test_results+='time=%.2f\n'%(end-begin)


                
######################
#                    #
#    Debut du RMS    #
#                    #
######################
        
if param['mode']=='rms_const_lr':
	begin=time.time()
        epochh = 0
	for epoch in xrange(param['nEpoch']):
                if decreasingLR:
                        epochh = epoch
                if maketest:
                        test_err_total=0.0
		        for index in xrange(test_num):
			        if len(train_index[index+900])>0:
				        test_err=rms_const_lr.rms(rnn,testX[index],testY[index],param['alpha'],False)
				        test_err_total+=test_err
                        test_err_total*=10
                        test_results+='|%f'%(test_err_total)
                        os.system('''echo '|%s' >> results_saved_%s_test_%d_%s_%.4f_%.4f_%.4f_%.4f_%.4f.log'''%(test_err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'],param['alpha']))
		
                err_total=0.0
		for index in xrange(train_num):
			if len(train_index[index])>0:
				err=rms_const_lr.rms(rnn,trainX[index],trainY[index],param['alpha'])
				err_total+=err
			if index%param['batch']==0:
				rnn.update(param['learnRates'],epochh+1)  
		rnn.update(param['learnRates'],epochh+1)
		print 'epoch %d completed!'%epoch
		if maketest:
                        print 'total train error=%f'%err_total
                        print 'total test error=%f'%test_err_total
                else:
		        print 'total error=%f'%err_total
		results+='|%f'%err_total
                os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'],param['alpha']))
	results+='\n'
	end=time.time()
	results+='time=%.2f\n'%(end-begin)

        if maketest:
                test_results+='\n'
	        test_results+='time=%.2f\n'%(end-begin)

                
                
                
with open(param['outfile'],param['outputMode']) as fopen:
	fopen.write(results)

if maketest:
        with open(param['test_outfile'],param['outputMode']) as fopen:
	        fopen.write(test_results)
