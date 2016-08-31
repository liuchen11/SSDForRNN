import sys
sys.path.insert(0,'util/')
sys.path.insert(0,'models/')
import numpy as np

import sharp

def const(model,learning_rate,decay=1.0):

    if model.buffer>0:

        model.dU/=model.buffer
        model.dW/=model.buffer
        model.dV/=model.buffer
        model.ds/=model.buffer
        model.gU/=model.buffer
        model.gW/=model.buffer
        model.gV/=model.buffer
        model.gs/=model.buffer

        model.update(learning_rate,decay)

def adagrad(model,learning_rate,decay=1.0,damping=1.0):

    if model.buffer>0:

        model.dU/=model.buffer
        model.dW/=model.buffer
        model.dV/=model.buffer
        model.ds/=model.buffer
        model.gU/=model.buffer
        model.gW/=model.buffer
        model.gV/=model.buffer
        model.gs/=model.buffer
        model.VU+=np.power(model.gU,2)
        model.VW+=np.power(model.gW,2)
        model.VV+=np.power(model.gV,2)
        model.Vs+=np.power(model.gs,2)
        model.DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
        model.DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
        model.DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
        model.Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)

        model.update(learning_rate,decay)

def rms(model,learning_rate,decay=1.0,alpha=0.9,damping=1.0):

    if model.buffer>0:

        model.gU/=model.buffer
        model.gW/=model.buffer
        model.gV/=model.buffer
        model.gs/=model.buffer
        model.dU/=model.buffer
        model.dW/=model.buffer
        model.dV/=model.buffer
        model.ds/=model.buffer

        model.VU=alpha*model.VU+(1-alpha)*np.power(model.gU,2)
        model.VW=alpha*model.VW+(1-alpha)*np.power(model.gW,2)
        model.VV=alpha*model.VV+(1-alpha)*np.power(model.gV,2)
        model.Vs=alpha*model.Vs+(1-alpha)*np.power(model.gs,2)
        model.DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
        model.DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
        model.DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
        model.Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)

        model.update(learning_rate,decay)
