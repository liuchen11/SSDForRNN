import sys
sys.path.insert(0,'../util/')
sys.path.insert(0,'../models/')
import math
import numpy as np

class optimizer(object):

    '''
    >>> Constructor
    '''
    def __init__(self,model):
        self.name='unspecified'

    '''
    >>> Update parameters according to specified learning rate
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    '''
    def move(self,model,learning_rates,decay=1.0):

        if model.buffer==0:
            return

        if type(learning_rates['U']) in [int, float]:
            learning_rates['U']=[learning_rates['U']]*(model.hidden_layers)
        if type(learning_rates['W']) in [int, float]:
            learning_rates['W']=[learning_rates['W']]*(model.hidden_layers)
        if type(learning_rates['s']) in [int, float]:
            learning_rates['s']=[learning_rates['s']]*(model.hidden_layers)

        for i in xrange(model.hidden_layers):
            model.U[i]-=model.dU[i]*learning_rates['U'][i]/math.sqrt(decay)
            model.dU[i]=np.zeros(model.dU[i].shape)
            model.gU[i]=np.zeros(model.gU[i].shape)
            model.W[i]-=model.dW[i]*learning_rates['W'][i]/math.sqrt(decay)
            model.dW[i]=np.zeros(model.dW[i].shape)
            model.gW[i]=np.zeros(model.gW[i].shape)
            model.s[i]-=model.ds[i]*learning_rates['s'][i]/math.sqrt(decay)
            model.ds[i]=np.zeros(model.ds[i].shape)
            model.gs[i]=np.zeros(model.gs[i].shape)
        model.V-=model.dV*learning_rates['V']/math.sqrt(decay)
        model.dV=np.zeros(model.dV.shape)
        model.gV=np.zeros(model.gV.shape)

        model.buffer=0


class constOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model):
        self.name='const'

    '''
    >>> optimize the network using constant learning rate
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    '''
    def update(self,model,learning_rates,decay=1.0):

        if model.buffer>0:

            for i in xrange(model.hidden_layers):
                model.gU[i]/=model.buffer
                model.dU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.dW[i]/=model.buffer
                model.gs[i]/=model.buffer
                model.ds[i]/=model.buffer

            model.gV/=model.buffer
            model.dV/=model.buffer

            self.move(model,learning_rates,decay)

class adagradOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model):
        self.name='adagrad'

        self.VU=[]
        self.DU=[]
        self.VW=[]
        self.DW=[]
        self.Vs=[]
        self.Ds=[]
        self.VV=np.zeros(model.V.shape)
        self.DV=np.zeros(model.V.shape)

        for i in xrange(model.hidden_layers):
            self.VU.append(np.zeros(model.V[i].shape))
            self.DU.append(np.zeros(model.V[i].shape))
            self.VW.append(np.zeros(model.W[i].shape))
            self.DW.append(np.zeros(model.W[i].shape))
            self.Vs.append(np.zeros(mdoel.s[i].shape))
            self.Ds.append(np.zeros(model.s[i].shape))

    '''
    >>> optimize the network using adagrad
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    >>> damping: float, optional. Damping coefficient
    '''
    def update(self,model,learning_rates,decay=1.0,damping=1.0):

        if model.buffer>0:

            for i in xrange(model.hidden_layers):

                model.dU[i]/=model.buffer
                model.dW[i]/=model.buffer
                model.ds[i]/=model.buffer
                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer
                self.VU[i]+=np.power(model.gU[i],2)
                self.VW[i]+=np.power(model.gW[i],2)
                self.Vs[i]+=np.power(model.gs[i],2)
                self.DU[i]=damping*np.ones(model.U[i].shape)+np.sqrt(self.VU[i])
                self.DW[i]=damping*np.ones(model.W[i].shape)+np.sqrt(self.VW[i])
                self.Ds[i]=damping*np.ones(model.s[i].shape)+np.sqrt(self.Vs[i])

            model.dV/=model.buffer
            model.gV/=model.buffer
            self.VV+=np.power(model.gV[i],2)
            self.DV=damping*np.ones(model.V.shape)+np.sqrt(self.VV)

            self.move(model,learning_rates,decay)

class rmspropOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model):
        self.name='rmsprop'

        self.VU=[]
        self.DU=[]
        self.VW=[]
        self.DW=[]
        self.Vs=[]
        self.Ds=[]
        self.VV=np.zeros(model.V.shape)
        self.DV=np.zeros(model.V.shape)

        for i in xrange(model.hidden_layers):
            self.VU.append(np.zeros(model.V[i].shape))
            self.DU.append(np.zeros(model.V[i].shape))
            self.VW.append(np.zeros(model.W[i].shape))
            self.DW.append(np.zeros(model.W[i].shape))
            self.Vs.append(np.zeros(mdoel.s[i].shape))
            self.Ds.append(np.zeros(model.s[i].shape))

    '''
    >>> optimize the network using rmsprop
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    >>> alpha: float, optional. parameter which correlates the memory and the new gradient
    >>> damping: float, optional. Damping coefficient
    '''
    def update(self,model,learning_rates,decay=1.0,alpha=0.9,damping=1.0):

        if model.buffer>0:

            for i in xrange(model.hidden_layers):

                model.dU[i]/=model.buffer
                model.dW[i]/=model.buffer
                model.ds[i]/=model.buffer
                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer
                self.VU[i]=alpha*self.VU[i]+(1-alpha)*np.power(model.gU[i],2)
                self.VW[i]=alpha*self.VW[i]+(1-alpha)*np.power(model.gW[i],2)
                self.Vs[i]=alpha*self.Vs[i]+(1-alpha)*np.power(model.gs[i],2)
                self.DU[i]=damping*np.ones(model.U[i].shape)+np.sqrt(self.VU[i])
                self.DW[i]=damping*np.ones(model.W[i].shape)+np.sqrt(self.VW[i])
                self.Ds[i]=damping*np.ones(model.s[i].shape)+np.sqrt(self.Vs[i])

            model.dV/=model.buffer
            model.gU/=model.buffer
            self.VV=alpha*self.VV+(1-alpha)*np.power(model.gV,2)
            self.DV=damping*np.ones(model.V.shape)+np.sqrt(self.VV)

            self.move(model,learning_rates,decay)



