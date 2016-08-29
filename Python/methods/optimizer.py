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
# '''
# >>> SGD using const learning rate
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# '''
# def sgd_const(model,learning_rate,decay=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.update(learning_rate,decay)

# '''
# >>> SSD using const learning_rate
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# '''

# def ssd_const(model,learning_rate,decay=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.gU=sharp.sharp(model.gU)
# 		model.gW=sharp.sharp(model.gW)

# 		model.update(learning_rate,decay)

# '''
# >>> SGD using adagrad
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# >>> damping: float, optional. damping coefficient used in adagrad
# '''
# def sgd_adagrad(model,learning_rate,decay=1.0,damping=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.VU+=np.power(model.gU,2)
# 		model.VW+=np.power(model.gW,2)
# 		model.VV+=np.power(model.gV,2)
# 		model.Vs+=np.power(model.gs,2)
# 		DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
# 		DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
# 		DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
# 		Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)

# 		model.gU=np.divide(model.gU,DU)
# 		model.gW=np.divide(model.gW,DW)
# 		model.gV=np.divide(model.gV,DV)
# 		model.gs=np.divide(model.gs,Ds)

# 		model.update(learning_rate,decay)

# '''
# >>> SSD using adagrad
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# >>> damping: float, optional. damping coefficient used in adagrad
# '''
# def ssd_adagrad(model,learning_rate,decay=1.0,damping=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.VU+=np.power(model.gU,2)
# 		model.VW+=np.power(model.gW,2)
# 		model.VV+=np.power(model.gV,2)
# 		model.Vs+=np.power(model.gs,2)
# 		DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
# 		DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
# 		DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
# 		Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)
# 		sqrtDU=np.sqrt(DU)
# 		sqrtDW=np.sqrt(DW)

# 		model.gU=np.divide(sharp.sharp(np.divide(model.gU,sqrtDU)),sqrtDU)
# 		model.gW=np.divide(sharp.sharp(np.divide(model.gW,sqrtDW)),sqrtDW)
# 		model.gV=np.divide(model.gV,DV)
# 		model.gs=np.divide(model.gs,Ds)

# 		model.update(learning_rate,decay)

# '''
# >>> SGD using RMSprop
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# >>> alpha: float, optional. parameter used in RMS which correlates the memory and the new gradient
# >>> damping: float, optional. damping coefficient used in adagrad
# '''
# def sgd_rms(model,learning_rate,decay=1.0,alpha=0.9,damping=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.VU=alpha*model.VU+(1-alpha)*np.power(model.gU,2)
# 		model.VW=alpha*model.VW+(1-alpha)*np.power(model.gW,2)
# 		model.VV=alpha*model.VV+(1-alpha)*np.power(model.gV,2)
# 		model.Vs=alpha*model.Vs+(1-alpha)*np.power(model.gs,2)
# 		DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
# 		DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
# 		DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
# 		Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)

# 		model.gU=np.divide(model.gU,DU)
# 		model.gW=np.divide(model.gW,DW)
# 		model.gV=np.divide(model.gV,DV)
# 		model.gs=np.divide(model.gs,Ds)

# 		model.update(learning_rate,decay)

# '''
# >>> SSD using RMSprop
# >>> model: RNN
# >>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
# >>> decay: float, optional. Learning rate decay coefficient.
# >>> alpha: float, optional. parameter used in RMS which correlates the memory and the new gradient
# >>> damping: float, optional. damping coefficient used in adagrad
# '''
# def ssd_rms(model,learning_rate,decay=1.0,alpha=0.9,damping=1.0):

# 	if model.buffer>0:

# 		model.gU/=model.buffer
# 		model.gW/=model.buffer
# 		model.gV/=model.buffer
# 		model.gs/=model.buffer

# 		model.VU=alpha*model.VU+(1-alpha)*np.power(model.gU,2)
# 		model.VW=alpha*model.VW+(1-alpha)*np.power(model.gW,2)
# 		model.VV=alpha*model.VV+(1-alpha)*np.power(model.gV,2)
# 		model.Vs=alpha*model.Vs+(1-alpha)*np.power(model.gs,2)
# 		DU=damping*np.ones(model.U.shape)+np.sqrt(model.VU)
# 		DW=damping*np.ones(model.W.shape)+np.sqrt(model.VW)
# 		DV=damping*np.ones(model.V.shape)+np.sqrt(model.VV)
# 		Ds=damping*np.ones(model.s.shape)+np.sqrt(model.Vs)
# 		sqrtDU=np.sqrt(DU)
# 		sqrtDW=np.sqrt(DW)

# 		model.gU=np.divide(sharp.sharp(np.divide(model.gU,sqrtDU)),sqrtDU)
# 		model.gW=np.divide(sharp.sharp(np.divide(model.gW,sqrtDW)),sqrtDW)
# 		model.gV=np.divide(model.gV,DV)
# 		model.gs=np.divide(model.gs,Ds)

# 		model.update(learning_rate,decay)
