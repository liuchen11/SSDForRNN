import sys
from math import sqrt
sys.path.insert(0,'../util/')

import gradient
import softmax
import numpy as np

class RNN(object):

	def __init__(self,input_size,hidden_size,output_size):
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.U=np.random.randn(hidden_size,input_size)*0.5
		self.W=np.random.randn(hidden_size,hidden_size)*0.5
		self.V=np.random.randn(output_size,hidden_size)*0.5
		self.s=np.random.randn(hidden_size)*0.5
		self.gU=np.zeros([hidden_size,input_size])
		self.gW=np.zeros([hidden_size,hidden_size])
		self.gV=np.zeros([output_size,hidden_size])
		self.gs=np.zeros([hidden_size])
		self.buffer=0

	def size(self):
		return self.input_size,self.hidden_size,self.output_size

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
		ret.buffer=self.buffer
		return ret

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

	def update(self,learning_rates,epoch=1):
		if self.buffer>0:
			self.U=self.U-self.gU*learning_rates['U']/(self.buffer*sqrt(epoch))
			self.W=self.W-self.gW*learning_rates['W']/(self.buffer*sqrt(epoch))
			self.V=self.V-self.gV*learning_rates['V']/(self.buffer*sqrt(epoch))
			self.s=self.s-self.gs*learning_rates['s']/(self.buffer*sqrt(epoch))

			self.buffer=0
			self.gU=np.zeros(self.gU.shape)
			self.gW=np.zeros(self.gW.shape)
			self.gV=np.zeros(self.gV.shape)
			self.gs=np.zeros(self.gs.shape)


