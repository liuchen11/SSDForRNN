import sys
import os
from random import shuffle,random
sys.path.insert(0,'models/')
sys.path.insert(0,'util/')
sys.path.insert(0,'methods/')

import RNN
import loader
import time
import grad
import optimizer
import numpy as np

'''
>>> Use RNN to train token labelling problem on dataset ATIS
>>> mode: string. Specify an optimizer like 'sgd_const'
>>> U_lr, W_lr, V_lr, s_lr: float. The learning rate for different parameter sets
>>> outfile: string. Set the output file we put the result in
'''

if len(sys.argv)!=8:
    print 'Usage: python atisLabel.py <mode> <U_lr> <W_lr> <V_lr> <s_lr> <a/w> <output_file>'
    exit(0)

maketest=True
decreasingLR=True
param={}
param['mode']=sys.argv[1]
param['trainXFile']='../atis/train_word.csv'            #Sentences in training set
param['trainYFile']='../atis/train_label.csv'           #Labels in training set
param['testXFile']='../atis/test_word.csv'              #Sentences in test set
param['testYFile']='../atis/test_label.csv'             #Labels in test set
param['dictFile']='../atis/dict10.csv'                  #Pretrained dict to map words to vectors
param['vectorDim']=100                                  #The dimension of word vectors
param['window']=3                                       #The length of the sliding windows
param['inputs']=param['vectorDim']*param['window']      #Input dimension
param['hiddens']=50                                     #Dimension of hidden state
param['outputs']=128                                    #Output dimension
param['batch']=25                                       #Batch size
param['nEpoch']=1000                                    #Number of training epochs
param['learnRates']={'U':float(sys.argv[2]),'W':float(sys.argv[3]),
    'V':float(sys.argv[4]),'s':float(sys.argv[5])}      #Learning rate of each parameter matrices
param['leftPad']=param['window']/2                      #Padding length
param['outputMode']=sys.argv[6]                         #Append or rewrite the output file
param['outfile']=sys.argv[7]                            #Output file
param['numberOfSave']=9                                 #The index of the saved file
param['alpha']=0.9                                      #A parameter used in RMSprop optimizer
param['damping']=1.0                                    #A parameter used in Adagrad or RMSprop optimizer
param['lrDecay']=False                                  #Enable or disable learning rate decay through epoch
param['useTest']=True                                   #Train only or train and test

train_index=loader.loadData(param['trainXFile'])
train_label=loader.loadData(param['trainYFile'])
test_index=loader.loadData(param['testXFile'])
test_label=load.loadData(param['testYFile'])
dictionary=loader.loadDict(param['dictFile'])
dictionary[-1]=np.random.randn(param['vectorDim'])*0.5

rnn=RNN.RNN(param['inputs'],param['hiddens'],param['outputs'])

########################################
#                                      #
#       Preprocessing the data         #
#                                      #
########################################

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

test_num=len(test_index)
testX=[]
testY=[]

for i in xrange(test_num):
    sentence_len=len(test_index[i])
    single_input=np.zeros([sentence_len,param['inputs']])
    single_label=np.zeros([sentence_len,param['outputs']])

    for p in xrange(sentence_len+param['window']-1):
        toFill=dictionary[-1]
        if p>=param['leftPad'] and p<sentence_len+param['leftPad']:
            word=test_index[i][p-param['leftPad']]
            if dictionary.has_key(word)==False:
                dictionary[word]=np.random.randn(param['vectorDim'])*0.05
            toFill=dictionary[word]
        startline=max(0,p-param['window']+1)
        endline=min(sentence_len-1,p)
        for line in range(startline,endline+1):
            col=p-line
            single_input[line,col*param['vectorDim']:(col+1)*param['vectorDim']]=toFill

    for w in xrange(sentence_len):
        single_label[w,test_label[i][w]]=1
    testX.append(single_input)
    testY.append(single_label)

results='''mode=%s,U_lr=%.4f,W_lr=%.4f,V_lr=%.4f,s_lr=%.4f\n'''%(param['mode'],param['learnRates']['U'],
    param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'])
print results,

######################
#                    #
#     Start SGD      #
#                    #
######################

if param['mode']=='sgd_const':
    begin=time.time()
    lr_decay=1
    for epoch in xrange(param['nEpoch']):
        if param['lrDecay']:
            lr_decay=epoch+1
        if param[useTest]:
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
                rnn.update(param['learnRates'],lr_decay)
        rnn.update(param['learnRates'],lr_decay)
        results+='|%f'%err_total
        os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
        print 'epoch %d completed!'%epoch
        if maketest:
            print 'total train error=%f'%err_total
            print 'total test error=%f'%test_err_total
        else:
            print 'total error=%f'%err_total
    results+='\n'
    end=time.time()
    results+='time=%.2f\n'%(end-begin)

    if maketest:
        test_results+='\n'
        test_results+='time=%.2f\n'%(end-begin)

######################
#                    #
#      Start SSD     #
#                    #
######################

if param['mode']=='ssd_const_lr':
    begin=time.time()
    lr_decay=1
    for epoch in xrange(param['nEpoch']):
        if decreasingLR:
            lr_decay=epoch+1
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
                rnn.update(param['learnRates'],lr_decay)
        rnn.update(param['learnRates'],lr_decay)
        results+='|%f'%err_total
        os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s']))
        print 'epoch %d completed!'%epoch
        if maketest:
            print 'total train error=%f'%err_total
            print 'total test error=%f'%test_err_total
        else:
            print 'total error=%f'%err_total
    results+='\n'
    end=time.time()
    results+='time=%.2f\n'%(end-begin)

    if maketest:
        test_results+='\n'
        test_results+='time=%.2f\n'%(end-begin)

######################
#                    #
#   Start RMSprop    #
#                    #
######################

if param['mode']=='ssd_rms':
    begin=time.time()
    lr_decay=1
    for epoch in xrange(param['nEpoch']):
        if decreasingLR:
            lr_decay=epoch+1
        if maketest:
            test_err_total=0.0
            for index in xrange(test_num):
                if len(train_index[index+900])>0:
                    test_err=ssd_rms.rms(rnn,testX[index],testY[index],param['alpha'],False)
                    test_err_total+=test_err
            test_err_total*=10
            test_results+='|%f'%(test_err_total)
            os.system('''echo '|%s' >> results_saved_%s_test_%d_%s_%.4f_%.4f_%.4f_%.4f_%.4f.log'''%(test_err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'],param['alpha']))

        err_total=0.0
        for index in xrange(train_num):
            if len(train_index[index])>0:
                err=ssd_rms.rms(rnn,trainX[index],trainY[index],param['alpha'])
                err_total+=err
            if index%param['batch']==0:
                rnn.update(param['learnRates'],epochh+1)  
        rnn.update(param['learnRates'],epochh+1)
        results+='|%f'%err_total
        os.system('''echo '|%s' >> results_saved_%s_train_%d_%s_%.4f_%.4f_%.4f_%.4f_%.4f.log'''%(err_total,param['dcrorcst'],param['numberOfSave'],param['mode'],param['learnRates']['U'],param['learnRates']['W'],param['learnRates']['V'],param['learnRates']['s'],param['alpha']))
        print 'epoch %d completed!'%epoch
        if maketest:
            print 'total train error=%f'%err_total
            print 'total test error=%f'%test_err_total
        else:
            print 'total error=%f'%err_total
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