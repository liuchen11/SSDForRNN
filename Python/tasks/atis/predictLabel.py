import sys
import os
sys.path.insert(0,'models/')
sys.path.insert(0,'util/')
sys.path.insert(0,'methods/')

import RNNs
import grads
import loader
import time
import optimizers
import xmlParser
import traceback
import numpy as np

def create_dir(path):
    parts=path.split(os.sep)
    pos=''
    for part in parts:
        pos+=part+os.sep
        if not os.path.exists(pos):
            try:
                os.mkdir(pos)
            except:
                print 'Failed to Create Folder %s'%pos

'''
>>> Use RNN to train token labelling problem on dataset ATIS
'''

if len(sys.argv)!=2:
    print 'Usage: python atisLabel.py <xml>'
    exit(0)

param=xmlParser.parse(sys.argv[1],flat=True)
neurons=[param['window']*param['vectorDim'],]+param['hiddens']+[param['outputs']]
learnRate={'W':param['learnRateW'],'U':param['learnRateU'],'V':param['learnRateV'],'s':param['learnRates']}
left_pad=param['window']/2

dictionary=loader.loadDict(param['dictFile'])
dictionary[-1]=np.random.randn(param['vectorDim'])*0.5
if param.has_key('modelSavedFolder'):
    create_dir(param['modelSavedFolder'])
if param.has_key('gradientSavedFile'):
    create_dir(os.path.dirname(param['gradientSavedFile']))

train_index=loader.loadData(param['trainXFile'])
train_label=loader.loadData(param['trainYFile'])
if param['trainOnly']==False:
    test_index=loader.loadData(param['testXFile'])
    test_label=loader.loadData(param['testYFile'])

rnn=RNNs.RNNs(neurons=neurons,nonlinearity=param['nonlinearity'])
results='''mode=%s,U_lr=%s,W_lr=%s,V_lr=%s,s_lr=%s,config file=%s\n'''%(param['mode'],
    param['learnRateU'],param['learnRateW'],param['learnRateV'],param['learnRates'],sys.argv[1])
print results,

#Preprocess the data
train_num=len(train_index)
trainX=[]
trainY=[]

for i in xrange(train_num):
    sentence_len=len(train_index[i])
    single_input=np.zeros([sentence_len,param['window']*param['vectorDim']])
    single_label=np.zeros([sentence_len,param['outputs']])

    for p in xrange(sentence_len+param['window']-1):
        toFill=dictionary[-1]
        if p>=left_pad and p<sentence_len+left_pad:
            word=train_index[i][p-left_pad]
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

if param['trainOnly']==False:

    test_num=len(test_index)
    testX=[]
    testY=[]

    for i in xrange(test_num):
        sentence_len=len(test_index[i])
        single_input=np.zeros([sentence_len,param['window']*param['vectorDim']])
        single_label=np.zeros([sentence_len,param['outputs']])

        for p in xrange(sentence_len+param['window']-1):
            toFill=dictionary[-1]
            if p>=left_pad and p<sentence_len+left_pad:
                word=test_index[i][p-left_pad]
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

#SGD
try:
    if param['mode']=='sgd_const':
        optimizer=optimizers.constOptimizer(rnn)
        begin=time.time()
        train_err_list=[]
        test_err_list=[]
        for epoch in xrange(param['nEpoch']):
            lr_decay=epoch if param['learnRateDecay'] else 1
            train_err=0.0
            for index in xrange(train_num):
                err=grads.sgd(rnn,trainX[index],trainY[index],optimizer)
                train_err+=err
                if index%param['batchSize']==0:
                    if param.has_key('gradientSavedFile'):
                        fopen=sys.stdout if param['gradientSavedFile']=='' else open(param['gradientSavedFile'], 'a')
                        rnn.print_gradient(fopen,notes={'epoch':epoch,'index':index})
                    optimizer.update(rnn,learnRate,decay=lr_decay)
            if param.has_key('gradientSavedFile'):
                fopen=sys.stdout if param['gradientSavedFile']=='' else open(param['gradientSavedFile'],'a')
                rnn.print_gradient(fopen,notes={'epoch':epoch,'index':index})
            optimizer.update(rnn,learnRate,decay=lr_decay)
            train_err_list.append(train_err)
            if param.has_key('modelSavedFolder'):
                rnn.save(param['modelSavedFolder']+os.sep+'sgd-epoch%d.pkl'%epoch)
            print 'train err on Epoch %d: %f'%(epoch,train_err)
            if param['trainOnly']==False:
                test_err=0.0
                for index in xrange(test_num):
                    err,_=rnn.forward(testX[index],testY[index])
                    test_err+=err
                test_err_list.append(test_err)
                print 'test err on Epoch %d: %f'%(epoch,test_err)
        end=time.time()

        for train_err_epoch in train_err_list:
            results+='|%f'%train_err_epoch
        results+='\n'
        if param['trainOnly']==False:
            for test_err_epoch in test_err_list:
                results+='|%f'%test_err_epoch
        results+='\n'
        results+='time=%.2f\n'%(end-begin)

    #SSD
    if param['mode']=='ssd_const':
        optimizer=optimizers.constOptimizer(rnn)
        begin=time.time()
        train_err_list=[]
        test_err_list=[]
        for epoch in xrange(param['nEpoch']):
            lr_decay=epoch if param['learnRateDecay'] else 1
            train_err=0.0
            for index in xrange(train_num):
                err=grads.ssd(rnn,trainX[index],trainY[index],optimizer)
                train_err+=err
                if index%param['batchSize']==0:
                    if param.has_key('gradientSavedFile'):
                        fopen=sys.stdout if param['gradientSavedFile']=='' else open(param['gradientSavedFile'], 'a')
                        rnn.print_gradient(fopen,notes={'epoch':epoch,'index':index})
                    optimizer.update(rnn,learnRate,decay=lr_decay)
            if param.has_key('gradientSavedFile'):
                fopen=sys.stdout if param['gradientSavedFile']=='' else open(param['gradientSavedFile'], 'a')
                rnn.print_gradient(fopen,notes={'epoch':epoch,'index':index})
            optimizer.update(rnn,learnRate,decay=lr_decay)
            train_err_list.append(train_err)
            if param.has_key('modelSavedFolder'):
                rnn.save(param['modelSavedFolder']+os.sep+'ssd-epoch%d.pkl'%epoch)
            print 'train err on Epoch %d: %f'%(epoch,train_err)
            if param['trainOnly']==False:
                test_err=0.0
                for index in xrange(test_num):
                    err,_=rnn.forward(testX[index],testY[index])
                    test_err+=err
                test_err_list.append(test_err)
                print 'test err on Epoch %d: %f'%(epoch,test_err)
        end=time.time()

        for train_err_epoch in train_err_list:
            results+='|%f'%train_err_epoch
        results+='\n'
        if param['trainOnly']==False:
            for test_err_epoch in test_err_list:
                results+='|%f'%test_err_epoch
        results+='\n'
        results+='time=%.2f\n'%(end-begin)
except:
    end=time.time()
    print 'An exception occurs'
    traceback.print_exc(sys.stdout)
    traceback.print_exc(open(param['outFile'],'a'))
    for train_err_epoch in train_err_list:
        results+='|%f'%train_err_epoch
    results+='\n'
    if param['trainOnly']==False:
        for test_err_epoch in test_err_list:
            results+='|%f'%test_err_epoch
    results+='\n'
    results+='time=%.2f\n'%(end-begin)
    if param.has_key('errOutputFolder'):
        create_dir(param['errOutputFolder'])
        rnn.save(param['errOutputFolder']+os.sep+'errModelSaved.pkl',testOnly=False)


with open(param['outFile'],'a') as fopen:
    fopen.write(results)







