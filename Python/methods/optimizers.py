import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
import math
import numpy as np

import sharp
import vectorNorm

'''
>>> mode: the type of the optimizer
>>> model: RNN or RNNs
>>> gradient_norm_limit: the maximum norm of the gradient allowed for each update
>>> ext_param: other parameters for some specific optimization method
'''
def constructOptimizer(mode,model,gradient_norm_limit=np.inf,ext_param={}):
    if mode.lower() in ['sgd_const','sgdconst']:
        print 'constructing a const sgd optimizer'
        return sgdConstOptimizer(model,gradient_norm_limit)
    elif mode.lower() in ['ssd_const','ssdconst']:
        print 'constructing a const ssd optimizer'
        return ssdConstOptimizer(model,gradient_norm_limit)
    elif mode.lower() in ['sgd_rmsprop','sgdrmsprop']:
        print 'constructing a rmsprop sgd optimizer'
        alpha=ext_param['alpha'] if ext_param.has_key('alpha') else 0.9
        damping=ext_param['damping'] if ext_param.has_key('damping') else 1.0
        return sgdRmspropOptimizer(model,gradient_norm_limit,alpha=alpha,damping=damping)
    elif mode.lower() in ['ssd_rmsprop','ssdrmsprop']:
        print 'constructing a rmsprop ssd optimizer'
        alpha=ext_param['alpha'] if ext_param.has_key('alpha') else 0.9
        damping=ext_param['damping'] if ext_param.has_key('damping') else 1.0
        return ssdRmspropOptimizer(model,gradient_norm_limit,alpha=alpha,damping=damping)
    else:
        print 'unrecognized name of the optimizer: %s'%mode
        print 'construting a general optimizer instead'
        return optimizer(model,gradient_norm_limit)

class optimizer(object):

    '''
    >>> Constructor
    '''
    def __init__(self,model,gradient_norm_limit=np.inf):
        self.name='unspecified'
        self.gradient_norm_limit=gradient_norm_limit

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
            if vectorNorm.norm(model.gU[i],2)*learning_rates['U'][i]>self.gradient_norm_limit:
                model.gU[i]=model.gU[i]/vectorNorm.norm(model.gU[i],2)/learning_rates['U'][i]*self.gradient_norm_limit
            if vectorNorm.norm(model.gW[i],2)*learning_rates['W'][i]>self.gradient_norm_limit:
                model.gW[i]=model.gW[i]/vectorNorm.norm(model.gW[i],2)/learning_rates['W'][i]*self.gradient_norm_limit
            if vectorNorm.norm(model.gs[i],2)*learning_rates['s'][i]>self.gradient_norm_limit:
                model.gs[i]=model.gs[i]/vectorNorm.norm(model.gs[i],2)/learning_rates['s'][i]*self.gradient_norm_limit
            model.U[i]-=model.gU[i]*learning_rates['U'][i]/math.sqrt(decay)
            model.W[i]-=model.gW[i]*learning_rates['W'][i]/math.sqrt(decay)
            model.s[i]-=model.gs[i]*learning_rates['s'][i]/math.sqrt(decay)
            model.gU[i]=np.zeros(model.gU[i].shape)
            model.gW[i]=np.zeros(model.gW[i].shape)
            model.gs[i]=np.zeros(model.gs[i].shape)

        if vectorNorm.norm(model.gV,2)*learning_rates['V']>self.gradient_norm_limit:
            model.gV=model.gV/vectorNorm.norm(model.gV,2)/learning_rates['V']*self.gradient_norm_limit
        model.V-=model.gV*learning_rates['V']/math.sqrt(decay)
        model.gV=np.zeros(model.gV.shape)

        model.buffer=0

class sgdConstOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model,gradient_norm_limit=np.inf):
        self.name='sgdConst'
        self.gradient_norm_limit=gradient_norm_limit

    '''
    >>> Optimize the network using constant learning rate
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    '''
    def update(self,model,learning_rates,decay=1.0):

        if model.buffer>0:

            for i in xrange(model.hidden_layers):
                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer

            model.gV/=model.buffer

            self.move(model,learning_rates,decay)

class ssdConstOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model,gradient_norm_limit=np.inf):
        self.name='ssdConst'
        self.gradient_norm_limit=gradient_norm_limit

    '''
    >>> Optimize the network using constant learning rate
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    '''
    def update(self,model,learning_rates,decay=1.0):

        if model.buffer>0:

            for i in xrange(model.hidden_layers):
                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer

                model.gU[i]=sharp.sharp(model.gU[i])
                model.gW[i]=sharp.sharp(model.gW[i])

            model.gV/=model.buffer

            self.move(model,learning_rates,decay)

class sgdRmspropOptimizer(optimizer):

    '''
    >>> Constructor
    '''
    def __init__(self,model,gradient_norm_limit,alpha,damping):
        self.name='sgdRmsprop'
        self.gradient_norm_limit=gradient_norm_limit
        self.alpha=alpha
        self.damping=damping

        self.VU=[]
        self.VW=[]
        self.Vs=[]

        for i in xrange(model.hidden_layers):
            self.VU.append(np.ones(model.U[i].shape))
            self.VW.append(np.ones(model.W[i].shape))
            self.Vs.append(np.ones(model.s[i].shape))

        self.VV=np.ones(model.V.shape)

    '''
    >>> optimize the network using rmspropones
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    >>> alpha: float, optional. parameter which correlates the memory and the new gradient
    >>> damping: float, optional. Damping coefficient
    '''
    def update(self,model,learning_rates,decay=1.0):

        if model.buffer!=0:

            for i in xrange(model.hidden_layers):
                
                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer

                DUi=np.ones(model.gU[i].shape)*self.damping+np.sqrt(self.VU[i])
                DWi=np.ones(model.gW[i].shape)*self.damping+np.sqrt(self.VW[i])
                Dsi=np.ones(model.gs[i].shape)*self.damping+np.sqrt(self.Vs[i])

                self.VU[i]=self.VU[i]*self.alpha+np.power(model.gU[i],2)*(1-self.alpha)
                self.VW[i]=self.VW[i]*self.alpha+np.power(model.gW[i],2)*(1-self.alpha)
                self.Vs[i]=self.Vs[i]*self.alpha+np.power(model.gs[i],2)*(1-self.alpha)

                model.gU[i]/=DUi
                model.gW[i]/=DWi
                model.gs[i]/=Dsi

            model.gV/=model.buffer
            DV=np.ones(model.gV.shape)*self.damping+np.sqrt(self.VV)
            self.VV=self.VV*self.alpha+np.power(model.gV,2)*(1-self.alpha)
            model.gV/=DV

            self.move(model,learning_rates,decay)

class ssdRmspropOptimizer(optimizer):

    def __init__(self,model,gradient_norm_limit,alpha,damping):
        self.name='ssdRmsprop'
        self.gradient_norm_limit=gradient_norm_limit
        self.alpha=alpha
        self.damping=damping

        self.VU=[]
        self.VW=[]
        self.Vs=[]

        for i in xrange(model.hidden_layers):
            self.VU.append(np.ones(model.U[i].shape))
            self.VW.append(np.ones(model.W[i].shape))
            self.Vs.append(np.ones(model.s[i].shape))

        self.VV=np.ones(model.V.shape)

    '''
    >>> optimize the network using rmspropones
    >>> model: RNN or RNNs
    >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
    >>> decay: float, optional. Learning rate decay coefficient.
    >>> alpha: float, optional. parameter which correlates the memory and the new gradient
    >>> damping: float, optional. Damping coefficient
    '''
    def update(self,model,learning_rates,decay=1.0):

        if model.buffer!=0:

            for i in xrange(model.hidden_layers):

                model.gU[i]/=model.buffer
                model.gW[i]/=model.buffer
                model.gs[i]/=model.buffer

                DUi=np.ones(model.gU[i].shape)*self.damping+np.sqrt(self.VU[i])
                DWi=np.ones(model.gW[i].shape)*self.damping+np.sqrt(self.VW[i])
                Dsi=np.ones(model.gs[i].shape)*self.damping+np.sqrt(self.Vs[i])

                self.VU[i]=self.VU[i]*self.alpha+np.power(model.gU[i],2)*(1-self.alpha)
                self.VW[i]=self.VW[i]*self.alpha+np.power(model.gW[i],2)*(1-self.alpha)
                self.Vs[i]=self.Vs[i]*self.alpha+np.power(model.gs[i],2)*(1-self.alpha)

                model.gU[i]=np.divide(sharp.sharp(np.divide(model.gU[i],np.sqrt(DUi))),np.sqrt(DUi))
                model.gW[i]=np.divide(sharp.sharp(np.divide(model.gW[i],np.sqrt(DWi))),np.sqrt(DWi))
                model.gs[i]=model.gs[i]/Dsi

            model.gV/=model.buffer
            DV=np.ones(model.gV.shape)*self.damping+np.sqrt(self.VV)
            self.VV=self.VV*self.alpha+np.power(model.gV,2)*(1-self.alpha)
            model.gV/=DV

            self.move(model,learning_rates,decay)



# class constOptimizer(optimizer):

#     '''
#     >>> Constructor
#     '''
#     def __init__(self,model,gradient_norm_limit=np.inf):
#         self.name='const'
#         self.gradient_norm_limit=gradient_norm_limit

#     '''
#     >>> optimize the network using constant learning rate
#     >>> model: RNN or RNNs
#     >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
#     >>> decay: float, optional. Learning rate decay coefficient.
#     '''
#     def update(self,model,learning_rates,decay=1.0):

#         if model.buffer>0:

#             for i in xrange(model.hidden_layers):
#                 model.gU[i]/=model.buffer
#                 model.dU[i]/=model.buffer
#                 model.gW[i]/=model.buffer
#                 model.dW[i]/=model.buffer
#                 model.gs[i]/=model.buffer
#                 model.ds[i]/=model.buffer

#             model.gV/=model.buffer
#             model.dV/=model.buffer

#             self.move(model,learning_rates,decay)

# class adagradOptimizer(optimizer):

#     '''
#     >>> Constructor
#     '''
#     def __init__(self,model,gradient_norm_limit=np.inf):
#         self.name='adagrad'
#         self.gradient_norm_limit=gradient_norm_limit

#         self.VU=[]
#         self.DU=[]
#         self.VW=[]
#         self.DW=[]
#         self.Vs=[]
#         self.Ds=[]
#         self.VV=np.ones(model.V.shape)
#         self.DV=np.ones(model.V.shape)

#         for i in xrange(model.hidden_layers):
#             self.VU.append(np.ones(model.U[i].shape))
#             self.DU.append(np.ones(model.U[i].shape))
#             self.VW.append(np.ones(model.W[i].shape))
#             self.DW.append(np.ones(model.W[i].shape))
#             self.Vs.append(np.ones(model.s[i].shape))
#             self.Ds.append(np.ones(model.s[i].shape))

#     '''
#     >>> optimize the network using adagrad
#     >>> model: RNN or RNNs
#     >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
#     >>> decay: float, optional. Learning rate decay coefficient.
#     >>> damping: float, optional. Damping coefficient
#     '''
#     def update(self,model,learning_rates,decay=1.0,damping=1.0):

#         if model.buffer>0:

#             for i in xrange(model.hidden_layers):

#                 model.dU[i]/=model.buffer
#                 model.dW[i]/=model.buffer
#                 model.ds[i]/=model.buffer
#                 model.gU[i]/=model.buffer
#                 model.gW[i]/=model.buffer
#                 model.gs[i]/=model.buffer
#                 self.VU[i]+=np.power(model.gU[i],2)
#                 self.VW[i]+=np.power(model.gW[i],2)
#                 self.Vs[i]+=np.power(model.gs[i],2)
#                 self.DU[i]=damping*np.ones(model.U[i].shape)+np.sqrt(self.VU[i])
#                 self.DW[i]=damping*np.ones(model.W[i].shape)+np.sqrt(self.VW[i])
#                 self.Ds[i]=damping*np.ones(model.s[i].shape)+np.sqrt(self.Vs[i])

#             model.dV/=model.buffer
#             model.gV/=model.buffer
#             self.VV+=np.power(model.gV[i],2)
#             self.DV=damping*np.ones(model.V.shape)+np.sqrt(self.VV)

#             self.move(model,learning_rates,decay)

# class rmspropOptimizer(optimizer):

#     '''
#     >>> Constructor
#     '''
#     def __init__(self,model,gradient_norm_limit=np.inf):
#         self.name='rmsprop'
#         self.gradient_norm_limit=gradient_norm_limit

#         self.VU=[]
#         self.DU=[]
#         self.VW=[]
#         self.DW=[]
#         self.Vs=[]
#         self.Ds=[]
#         self.VV=np.ones(model.V.shape)
#         self.DV=np.ones(model.V.shape)

#         for i in xrange(model.hidden_layers):
#             self.VU.append(np.ones(model.U[i].shape))
#             self.DU.append(np.ones(model.U[i].shape))
#             self.VW.append(np.ones(model.W[i].shape))
#             self.DW.append(np.ones(model.W[i].shape))
#             self.Vs.append(np.ones(model.s[i].shape))
#             self.Ds.append(np.ones(model.s[i].shape))

#     '''
#     >>> optimize the network using rmspropones>>> model: RNN or RNNs
#     >>> learning_rates: dict[str->float/list[float]], learning rate for each parameter
#     >>> decay: float, optional. Learning rate decay coefficient.
#     >>> alpha: float, optional. parameter which correlates the memory and the new gradient
#     >>> damping: float, optional. Damping coefficient
#     '''
#     def update(self,model,learning_rates,decay=1.0,alpha=0.9,damping=1.0):

#         if model.buffer>0:

#             for i in xrange(model.hidden_layers):

#                 model.dU[i]/=model.buffer
#                 model.dW[i]/=model.buffer
#                 model.ds[i]/=model.buffer
#                 model.gU[i]/=model.buffer
#                 model.gW[i]/=model.buffer
#                 model.gs[i]/=model.buffer
#                 self.VU[i]=alpha*self.VU[i]+(1-alpha)*np.power(model.gU[i],2)
#                 self.VW[i]=alpha*self.VW[i]+(1-alpha)*np.power(model.gW[i],2)
#                 self.Vs[i]=alpha*self.Vs[i]+(1-alpha)*np.power(model.gs[i],2)
#                 self.DU[i]=damping*np.ones(model.U[i].shape)+np.sqrt(self.VU[i])
#                 self.DW[i]=damping*np.ones(model.W[i].shape)+np.sqrt(self.VW[i])
#                 self.Ds[i]=damping*np.ones(model.s[i].shape)+np.sqrt(self.Vs[i])

#             model.dV/=model.buffer
#             model.gV/=model.buffer
#             self.VV=alpha*self.VV+(1-alpha)*np.power(model.gV,2)
#             self.DV=damping*np.ones(model.V.shape)+np.sqrt(self.VV)

#             self.move(model,learning_rates,decay)
