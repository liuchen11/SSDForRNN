import sys
import copy
import cPickle
import numpy as np
from math import sqrt
sys.path.insert(0,'util/')

import softmax
import gradient
import matrixNorm

'''
>>> multi-layer Recurrent Neural Network
'''
class RNNs(object):

    '''
    >>> Constructor
    >>> neurons: a list of integers representing the number of neurons in each layer
    >>> self.gU, self.gW, self.gV: gradient of the corresponding parameters
    >>> self.buffer: the number of processed instances within a batch
    '''
    def __init__(self,neurons,nonlinearity):
        self.layers=len(neurons)
        self.nonlinearity=nonlinearity
        self.hidden_layers=self.layers-2
        self.input_size=neurons[0]
        self.hidden_size=neurons[1:-1]
        self.output_size=neurons[-1]
        self.size=neurons

        self.U=[]
        self.gU=[]
        self.W=[]
        self.gW=[]
        self.s=[]
        self.gs=[]
        for i in xrange(self.hidden_layers):
            Ui=np.random.randn(neurons[i+1],neurons[i])*0.5
            gUi=np.zeros(Ui.shape)
            self.U.append(Ui)
            self.gU.append(gUi)

            Wi=np.random.randn(neurons[i+1],neurons[i+1])*0.5
            Wi_sinf_norm=matrixNorm.norm(Wi,np.inf)
            Wi/=Wi_sinf_norm
            gWi=np.zeros(Wi.shape)
            self.W.append(Wi)
            self.gW.append(gWi)

            si=np.random.randn(neurons[i+1])*0.5
            gsi=np.zeros(si.shape)
            self.s.append(si)
            self.gs.append(gsi)

        self.V=np.random.randn(neurons[-1],neurons[-2])*0.5
        self.gV=np.zeros(self.V.shape)

        self.activation=[]
        self.dactivation=[]
        for func in nonlinearity:
            if func.lower() in ['relu',]:
                self.activation.append(gradient.relu)
                self.dactivation.append(gradient.drelu)
            elif func.lower() in ['sigmoid','sigd','sigmd']:
                self.activation.append(gradient.sigmoid)
                self.dactivation.append(gradient.dsigmoid)
            elif func.lower() in ['tanh']:
                self.activation.append(gradient.tanh)
                self.dactivation.append(gradient.dtanh)
            else:
                raise Exception('unrecognized function name %s'%func)

        self.params=self.U+self.W+self.s+[self.V,]
        self.buffer=0

    '''
    >>> make a deep copy of current network, including the size and the parameters
    '''
    def copy(self):
        ret=RNNs(self.size,self.nonlinearity)

        for i in xrange(self.hidden_layers):
            ret.U[i]=np.copy(self.U[i])
            ret.gU[i]=np.copy(self.gU[i])
            ret.W[i]=np.copy(self.W[i])
            ret.gW[i]=np.copy(self.gW[i])
            ret.s[i]=np.copy(self.s[i])
            ret.gs[i]=np.copy(self.gs[i])

        ret.V=np.copy(self.V)
        ret.gV=np.copy(self.gV)

        ret.params=ret.U+ret.W+ret.s+[ret.V,]
        ret.buffer=self.buffer
        return ret

    '''
    >>> Given a squence and its ground_truth, calculate the prediction of the network and the corresponding loss
    >>> states: Input states. 2-D array of shape S*N where S is the length of the squence and N the input dimension
    >>> ground_truth: Labels. 2-D array of S*H or 1*H depending on final_label where H is the dimension of the output
    >>> final_label: Boolean. If True, the sequence only have a final label otherwise each token has a corresponding label
    '''
    def forward(self,states,ground_truth,final_label=False):
        err=0.0
        outputs=[]
        num=len(states)

        hidden_states=copy.copy(self.s)

        for idx,token in enumerate(states):
            hidden_states[0]=self.activation[0](np.dot(self.U[0],token)+np.dot(self.W[0],hidden_states[0]))
            for i in xrange(1,self.hidden_layers):
                hidden_states[i]=self.activation[0](np.dot(self.U[i],hidden_states[i-1])+np.dot(self.W[i],hidden_states[i]))
            if idx==len(states)-1 or not final_label:
                proj=np.dot(self.V,hidden_states[-1])
                soft=softmax.softmax(proj)
                logsoft=np.log(soft)

                err-=np.dot(ground_truth[0],logsoft) if final_label else np.dot(ground_truth[idx],logsoft)
                outputs.append(soft)
        
        err=err if final_label else err/num
        return err,outputs


    '''
    >>> print the gradient information
    >>> out_file: file in mode 'w'
    >>> notes: dict, extra information need to be mentioned
    '''
    def print_gradient(self,out_file,notes={}):
        out_file.write('===========================================\n')
        for item in notes:
            out_file.write(str(item)+':'+str(notes[item])+'\n')
        if self.buffer==0:
            out_file.write('This is no gradient information yet\n')
            return    
        out_file.write('This is a recurrent neural network of %d hidden layer(s)\n'%self.hidden_layers)
        for idx in xrange(self.hidden_layers):
            out_file.write('Average abs value of matrix gU in layer %d: %.5f\n'%(idx+1, np.mean(np.abs(self.gU[idx]))/self.buffer))
            out_file.write('Average abs value of matrix gW in layer %d: %.5f\n'%(idx+1, np.mean(np.abs(self.gW[idx]))/self.buffer))
            out_file.write('Average abs value of matrix gs in layer %d: %.5f\n'%(idx+1, np.mean(np.abs(self.gs[idx]))/self.buffer))
        out_file.write('Average abs value of matrix gV: %.5f\n'%(np.mean(np.abs(self.gV))/self.buffer))
        out_file.flush()
            
    '''
    >>> save the model
    >>> out_file: str, name of saved file
    >>> testOnly: if True, only parameters are saved; otherwise, all information will be saved
    '''
    def save(self,out_file,testOnly=True):
        if testOnly:
            cPickle.dump({'size':self.size,'U':self.U,'W':self.W,'V':self.V,'s':self.s},
                open(out_file,'wb'))
        else:
            cPickle.dump({'size':self.size,'U':self.U,'W':self.W,'V':self.V,'s':self.s,
                'gU':self.gU,'gW':self.gW,'gV':self.gV,'gs':self.gs},open(out_file,'wb'))
        print 'parameters saved in file: %s'%out_file

    '''
    >>> load the parameters
    >>> in_file: str, name of the file to load
    >>> testOnly: if True, only load connections; otherwise, load all information
    '''
    def load(self,in_file,testOnly=True):
        info=cPickle.load(open(in_file,'rb'))
        if info['size']!=self.size:
            print 'the model\' size is different from the loaded file!'
            print 'the size of the model: ',self.size
            print 'the size information in config file: ',info['size']
            return
        self.U=info['U'];self.W=info['W'];self.V=info['V'];self.s=info['s']
        if testOnly==False:
            self.gU=info['gU'];self.gW=info['gW'];self.gV=info['gV'];self.gs=info['gs']
        print 'parameters loaded from file: %s'%in_file
        
        