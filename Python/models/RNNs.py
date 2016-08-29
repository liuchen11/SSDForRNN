import sys
sys.path.insert(0,'util/')

from math import sqrt
import cPickle
import gradient
import softmax
import numpy as np
import copy

'''
>>> multi-layer Recurrent Neural Network
'''
class RNNs(object):

    '''
    >>> Constructor
    >>> neurons: a list of integers representing the number of neurons in each layer
    >>> self.gU, self.gW, self.gV: gradient of the corresponding parameters
    >>> self.dU, self.dW, self.gV: the direction of parameters' update
    >>> self.buffer: the number of processed instances within a batch
    '''
    def __init__(self,neurons):
        self.layers=len(neurons)
        self.hidden_layers=self.layers-2
        self.input_size=neurons[0]
        self.hidden_size=neurons[1:-1]
        self.output_size=neurons[-1]
        self.size=neurons

        self.U=[]
        self.gU=[]
        self.dU=[]
        self.W=[]
        self.gW=[]
        self.dW=[]
        self.s=[]
        self.gs=[]
        self.ds=[]
        for i in xrange(self.hidden_layers):
            Ui=np.random.randn(neurons[i+1],neurons[i])*0.5
            gUi=np.zeros(Ui.shape)
            dUi=np.zeros(Ui.shape)
            self.U.append(Ui)
            self.gU.append(gUi)
            self.dU.append(dUi)

            Wi=np.random.randn(neurons[i+1],neurons[i+1])*0.5
            gWi=np.zeros(Wi.shape)
            dWi=np.zeros(Wi.shape)
            self.W.append(Wi)
            self.gW.append(gWi)
            self.dW.append(dWi)

            si=np.random.randn(neurons[i+1])*0.5
            gsi=np.zeros(si.shape)
            dsi=np.zeros(si.shape)
            self.s.append(si)
            self.gs.append(gsi)
            self.ds.append(dsi)

        self.V=np.random.randn(neurons[-1],neurons[-2])*0.5
        self.gV=np.zeros(self.V.shape)
        self.dV=np.zeros(self.V.shape)

        self.params=self.U+self.W+self.s+[self.V,]
        self.buffer=0

    '''
    >>> make a deep copy of current network, including the size and the parameters
    '''
    def copy(self):
        ret=RNNs(self.size)

        for i in xrange(self.hidden_layers):
            ret.U[i]=np.copy(self.U[i])
            ret.gU[i]=np.copy(self.gU[i])
            ret.dU[i]=np.copy(self.dU[i])
            ret.W[i]=np.copy(self.W[i])
            ret.gW[i]=np.copy(self.gW[i])
            ret.dW[i]=np.copy(self.dW[i])
            ret.s[i]=np.copy(self.s[i])
            ret.gs[i]=np.copy(self.gs[i])
            ret.ds[i]=np.copy(self.ds[i])

        ret.V=np.copy(self.V)
        ret.gV=np.copy(self.gV)
        ret.dV=np.copy(self.dV)

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
            hidden_states[0]=gradient.sigmoid(np.dot(self.U[0],token)+np.dot(self.W[0],hidden_states[0]))
            for i in xrange(1,self.hidden_layers):
                hidden_states[i]=gradient.sigmoid(np.dot(self.U[i],hidden_states[i-1])+np.dot(self.W[i],hidden_states[i]))
            if idx==len(states)-1 or not final_label:
                proj=np.dot(self.V,hidden_states[-1])
                soft=softmax.softmax(proj)
                logsoft=np.log(soft)

                err-=np.dot(ground_truth[0],logsoft) if final_label else np.dot(ground_truth[idx],logsoft)
                outputs.append(soft)
        
        err=err if final_label else err/num
        return err,outputs

    '''
    >>> Update parameters according to specified learning rate
    >>> learning_rate: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    '''
    def update(self,learning_rates,decay=1.0):
        if self.buffer==0:
            return
        
        if type(learning_rates['U'])==int:
            learning_rates['U']=[learning_rates['U']]*(self.hidden_layers)
        if type(learning_rates['W'])==int:
            learning_rates['W']=[learning_rates['W']]*(self.hidden_layers)
        if type(learning_rates['s'])==int:
            learning_rates['s']=[learning_rates['s']]*(self.hidden_layers)

        for i in xrange(self.hidden_layers):
            self.U[i]-=self.dU[i]*learning_rates['U'][i]/sqrt(decay)
            self.dU[i]=np.zeros(self.dU[i].shape)
            self.gU[i]=np.zeros(self.gU[i].shape)
            self.W[i]-=self.dW[i]*learning_rates['W'][i]/sqrt(decay)
            self.dW[i]=np.zeros(self.dW[i].shape)
            self.gW[i]=np.zeros(self.gW[i].shape)
            self.s[i]-=self.ds[i]*learning_rates['s'][i]/sqrt(decay)
            self.ds[i]=np.zeros(self.ds[i].shape)
            self.gs[i]=np.zeros(self.gs[i].shape)
        self.V-=self.dV*learning_rates['V']/sqrt(decay)
        self.dV=np.zeros(self.dV.shape)
        self.gV=np.zeros(self.gV.shape)

        self.buffer=0

    '''
    >>> save the model
    >>> out_file: str, name of saved file
    >>> testOnly: if True, only parameters are saved; otherwise, all information will be saved.
    '''
    def save(self,out_file,testOnly=True):
        if testOnly:
            cPickle.dump({'size':self.size,'U':self.U,'W':self.W,'V':self.V},
                open(out_file,'wb'))
        else:
            cPickle.dump(self,open(out_file,'wb'))
