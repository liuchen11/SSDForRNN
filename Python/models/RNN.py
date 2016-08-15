import sys
from math import sqrt
sys.path.insert(0,'../util/')

import cPickle
import gradient
import softmax
import numpy as np

'''
>>> 1-layer Recurrent Neural Network model
'''
class RNN(object):

	'''
	>>> Constructor
	>>> self.gU,self.gW,self.gV,self.gs: the gradient of the corresponding parameters
	>>> Parameters are initialized according to a Gussian distribution
	>>> self.buffer: int. Number of buffered training instance. We flush the buffer when this number meets the size of batch
	>>> self.VU,self.VW,self.VV,self.Vs: Matrix of the same shape as the corresponding parameters. Used to control the magnitude of the learning rate in RMS and Adagrad.
	'''
	def __init__(self,input_size,hidden_size,output_size):
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.U=np.random.randn(hidden_size,input_size)*0.5
		self.W=np.random.randn(hidden_size,hidden_size)*0.5
		self.V=np.random.randn(output_size,hidden_size)*0.5
		self.s=np.random.randn(hidden_size)*0.5
		self.gU=np.zeros(self.U.shape)
		self.gW=np.zeros(self.W.shape)
		self.gV=np.zeros(self.V.shape)
		self.gs=np.zeros(self.s.shape)
		self.dU=np.zeros(self.U.shape)
		self.dW=np.zeros(self.W.shape)
		self.dV=np.zeros(self.V.shape)
		self.ds=np.zeros(self.s.shape)
		self.VU=np.zeros(self.U.shape)
		self.VW=np.zeros(self.W.shape)
		self.VV=np.zeros(self.V.shape)
		self.Vs=np.zeros(self.s.shape)
		self.DU=np.ones(self.U.shape)
		self.DW=np.ones(self.W.shape)
		self.DV=np.ones(self.V.shape)
		self.Ds=np.ones(self.s.shape)
		self.buffer=0

	'''
	>>> Get the size of the network
	'''
	def size(self):
		return self.input_size,self.hidden_size,self.output_size

	'''
	>>> Make a deep copy of current network, including the size and the parameters
	'''
	def copy(self):
		ret=RNN(self.input_size,self.hidden_size,self.output_size)
		ret.U=np.copy(self.U)
		ret.W=np.copy(self.W)
		ret.V=np.copy(self.V)
		ret.s=np.copy(self.s)
		ret.gU=np.copy(self.gU)
		ret.gW=np.copy(self.gW)
		ret.gV=np.copy(self.gV)
		ret.gs=np.copy(self.gs)
		ret.dU=np.copy(self.dU)
		ret.dW=np.copy(self.dW)
		ret.dV=np.copy(self.dV)
		ret.ds=np.copy(self.ds)
		ret.VU=np.copy(self.VU)
		ret.VW=np.copy(self.VW)
		ret.VV=np.copy(self.VV)
		ret.Vs=np.copy(self.Vs)
		ret.DU=np.copy(self.DU)
		ret.DW=np.copy(self.DW)
		ret.DV=np.copy(self.DV)
		ret.Ds=np.copy(self.Ds)
		ret.buffer=self.buffer
		return ret

	'''
	>>> Given a sequence and its ground truth, calculate the prediction of the network and corresponding loss
	>>> states: Input states. 2-D array of shape S*N where S is the sequence length and N is the input dimension
	>>> ground_truth: Labels. 2-D array of shape S*H where S is the sequence length and H is the number of labels
	'''
	def runTokens(self,states,ground_truth):
		err=0.0
		outputs=[]
		num=len(states)
		hidden_states=self.s

		for i in xrange(num):
			input_part=np.dot(self.U,states[i])
			recur_part=np.dot(self.W,hidden_states)
			hidden_states=gradient.sigmoid(input_part+recur_part)
			proj=np.dot(self.V,hidden_states)
			soft=softmax.softmax(proj)
			logsoft=np.log(soft)

			err=err-np.dot(ground_truth[i],logsoft)
			outputs.append(soft)

		err=err/num
		return err,outputs

	'''
	>>> Update parameters according to specified learning rates
	>>> learning_rate: dict[string -> float]. learning_rate['U'] means the learning rate of matrix U and so on.
	>>> decay: float, optional. Learning rate decay coefficient.
	'''
	def update(self,learning_rates,decay=1):

		self.U=self.U-self.dU*learning_rates['U']/sqrt(decay)
		self.W=self.W-self.dW*learning_rates['W']/sqrt(decay)
		self.V=self.V-self.dV*learning_rates['V']/sqrt(decay)
		self.s=self.s-self.ds*learning_rates['s']/sqrt(decay)

		self.buffer=0
		self.gU=np.zeros(self.gU.shape)
		self.gW=np.zeros(self.gW.shape)
		self.gV=np.zeros(self.gV.shape)
		self.gs=np.zeros(self.gs.shape)
		self.dU=np.zeros(self.dU.shape)
		self.dW=np.zeros(self.dW.shape)
		self.dV=np.zeros(self.dV.shape)
		self.ds=np.zeros(self.ds.shape)

	'''
	>>> Return the Euclidean norm (steps) of the gradient of all parameters
	'''
	def gradient_steps(self,learning_rates):
			vec_gU=self.gU.reshape(self.gU.size)*learning_rates['U']
			vec_gW=self.gW.reshape(self.gW.size)*learning_rates['W']
			vec_gV=self.gV.reshape(self.gV.size)*learning_rates['V']
			vec_gs=self.gs.reshape(self.gs.size)*learning_rates['s']
			vec_grad=np.concatenate([vec_gU,vec_gW,vec_gV,vec_gs],axis=0)
			return np.linalg.norm(vec_grad,2)

	'''
	>>> save the model
	'''
	def save(self,out_file):
		cPickle.dump([self.U,self.W,self.V,self.s],open(out_file,'w'))
