import os
import sys
import time
import traceback
import numpy as np
sys.path.insert(0,'models/')
sys.path.insert(0,'util/')
sys.path.insert(0,'methods/')

import RNNs
import grads
import loader
import optimizers
import xmlParser
import matrixNorm

'''
>>> Use RNN to train token labelling problem on dataset ATIS
'''

if len(sys.argv)!=2:
    print 'Usage: python atisLabel.py <xml>'
    exit(0)

param=xmlParser.parse(sys.argv[1],flat=True)

# Neural Network
vector_dim=param['vectorDim']
window_size=param['window']
left_pad=window_size/2
input_size=window_size*vector_dim
hidden_neurons=param['hiddens']
output_size=param['outputs']
neurons=[input_size,]+hidden_neurons+[output_size]
batch_size=param['batchSize']
nonlinearity=param['nonlinearity']
model2load=param['model2load'] if param.has_key('model2load') and os.path.exists(param['model2load']) else None
n_epoch=param['nEpoch']

# Dataset
dict_file=param['dictFile']
trainXFile=param['trainXFile']
trainYFile=param['trainYFile']
train_only=param['trainOnly']
testXFile=param['testXFile'] if train_only==False else None
testYFile=param['testYFile'] if train_only==False else None

# Optimization Method
learn_rate={'W':param['learnRateW'],'U':param['learnRateU'],'V':param['learnRateV'],'s':param['learnRates']}
gradient_upper_bound=np.inf if not param.has_key('gradientThreshold') else param['gradientThreshold']
learn_rate_decay=param['learnRateDecay']
mode=param['mode']
ext_param={}
if param.has_key('alpha'):
    ext_param['alpha']=param['alpha']
if param.has_key('damping'):
    ext_param['damping']=param['damping']

# IO
name='.'.join(sys.argv[1].split(os.sep)[-1].split('.')[:-1]) if not param.has_key('name') else param['name']
err_output_folder=param['errOutputFolder']
model_saved_folder=param['modelSavedFolder']
gradient_saved_folder=param['gradientSavedFolder'] if param.has_key('gradientSavedFolder') else None
out_file=param['outFile']

dictionary=loader.loadDict(dict_file)
dictionary[-1]=np.random.randn(vector_dim)*0.5
if not os.path.exists(model_saved_folder):
    os.makedirs(model_saved_folder)
if gradient_saved_folder!=None:
    if not os.path.exists(gradient_saved_folder):
        os.makedirs(gradient_saved_folder)

train_index=loader.loadData(trainXFile)
train_label=loader.loadData(trainYFile)
if train_only==False:
    test_index=loader.loadData(testXFile)
    test_label=loader.loadData(testYFile)

rnn=RNNs.RNNs(neurons=neurons,nonlinearity=nonlinearity)
if model2load!=None:
    print 'load weights from file: %s'%model2load
    rnn.load(model2load,testOnly=True)

results='''mode=%s,U_lr=%s,W_lr=%s,V_lr=%s,s_lr=%s,config file=%s\n'''%(mode,
    str(learn_rate['U']),str(learn_rate['W']),str(learn_rate['V']),str(learn_rate['s']),sys.argv[1])
print results,

#Preprocess the data
train_num=len(train_index)
trainX=[]
trainY=[]

for i in xrange(train_num):
    sentence_len=len(train_index[i])
    single_input=np.zeros([sentence_len,input_size])
    single_label=np.zeros([sentence_len,output_size])

    for p in xrange(sentence_len+window_size-1):
        toFill=dictionary[-1]
        if p>=left_pad and p<sentence_len+left_pad:
            word=train_index[i][p-left_pad]
            if dictionary.has_key(word)==False:
                print 'Unrecognized Word: %s'%word
                dictionary[word]=np.random.randn(vector_dim)*0.05
            toFill=dictionary[word]
        startline=max(0,p-window_size+1)
        endline=min(sentence_len-1,p)
        for line in range(startline,endline+1):
            col=p-line
            single_input[line,col*vector_dim:(col+1)*vector_dim]=toFill

    for w in xrange(sentence_len):
        single_label[w,train_label[i][w]]=1
    trainX.append(single_input)
    trainY.append(single_label)

if train_only==False:

    test_num=len(test_index)
    testX=[]
    testY=[]

    for i in xrange(test_num):
        sentence_len=len(test_index[i])
        single_input=np.zeros([sentence_len,input_size])
        single_label=np.zeros([sentence_len,output_size])

        for p in xrange(sentence_len+window_size-1):
            toFill=dictionary[-1]
            if p>=left_pad and p<sentence_len+left_pad:
                word=test_index[i][p-left_pad]
                if dictionary.has_key(word)==False:
                    dictionary[word]=np.random.randn(vector_dim)*0.05
                toFill=dictionary[word]
            startline=max(0,p-window_size+1)
            endline=min(sentence_len-1,p)
            for line in range(startline,endline+1):
                col=p-line
                single_input[line,col*vector_dim:(col+1)*vector_dim]=toFill

        for w in xrange(sentence_len):
            single_label[w,test_label[i][w]]=1
        testX.append(single_input)
        testY.append(single_label)

try:
    optimizer=optimizers.constructOptimizer(mode,rnn,gradient_upper_bound,ext_param=ext_param)
    begin=time.time()
    train_err_list=[]
    test_err_list=[]
    norm=matrixNorm.norm(rnn.W[0],np.inf)
    rnn.W[0]/=norm
    for epoch in xrange(n_epoch):
        lr_decay=epoch if learn_rate_decay else 1
        train_err_this_epoch=[]
        for index in xrange(train_num):
            err=grads.gradient(rnn,trainX[index],trainY[index])
            print 'Epoch = %d, index = %d, Error = %.4f\r'%(epoch,index,err),
            train_err_this_epoch.append(err)
            if (index+1)%batch_size==0:
                print 'Average Error in Epoch = %d, index = [%d, %d): %.4f'%(epoch, index+1-batch_size, index+1, np.mean(train_err_this_epoch[-batch_size:]))
                fopen=sys.stdout if gradient_saved_folder==None else gradient_saved_folder+os.sep+'%s.pkl'%name
                rnn.save_gradient(fopen,notes={'epoch':epoch,'index':index})
                optimizer.update(rnn,learn_rate,decay=lr_decay)
                # norm=matrixNorm.norm(rnn.W[0],np.inf)
                # rnn.W[0]/=norm
        fopen=sys.stdout if gradient_saved_folder==None else gradient_saved_folder+os.sep+'%s.pkl'%name
        rnn.save_gradient(fopen,notes={'epoch':epoch,'index':index})
        optimizer.update(rnn,learn_rate,decay=lr_decay)
        train_err=np.mean(train_err_this_epoch)
        train_err_list.append(train_err)
        rnn.save(model_saved_folder+os.sep+'%s-epoch-%d.pkl'%(name, epoch))
        print 'train err on Epoch %d: %f'%(epoch,train_err)
        if train_only==False:
            test_err_this_epoch=[]
            for index in xrange(test_num):
                err,_=rnn.forward(testX[index],testY[index])
                test_err_this_epoch.append(err)
                print 'Test, index = %d, Error = %.4f, Average Error = %.4f\r'%(index, err, np.mean(test_err_this_epoch)),
            test_err_list.append(np.mean(test_err_this_epoch))
            print 'test err on Epoch %d: %f'%(epoch,np.mean(test_err_this_epoch))
    end=time.time()

    for train_err_epoch in train_err_list:
        results+='|%f'%train_err_epoch
    results+='\n'
    if train_only==False:
        for test_err_epoch in test_err_list:
            results+='|%f'%test_err_epoch
    results+='\n'
    results+='time=%.2f\n'%(end-begin)

except:
    end=time.time()
    print 'An exception occurs'
    traceback.print_exc(sys.stdout)
    traceback.print_exc(open(out_file,'a'))
    for train_err_epoch in train_err_list:
        results+='|%f'%train_err_epoch
    results+='\n'
    if train_only==False:
        for test_err_epoch in test_err_list:
            results+='|%f'%test_err_epoch
    results+='\n'
    results+='time=%.2f\n'%(end-begin)
    if not os.path.exists(err_output_folder):
        os.makedirs(err_output_folder)
    rnn.save(err_output_folder+os.sep+'%s-errModelSaved.pkl'%name,testOnly=False)

with open(out_file,'a') as fopen:
    fopen.write(results)







