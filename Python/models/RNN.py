import sys
sys.path.insert(0,'../util/')

import gradient
import softmax
import numpy as np

class RNN(object):

	def __init__(self,input_size,hidden_size,output_size):
		# self.U=np.random.random([hidden_size,input_size])*2-1
		# self.W=np.random.random([hidden_size,hidden_size])*2-1
		# self.V=np.random.random([output_size,hidden_size])*2-1
		# self.s=np.random.random([hidden_size])*2-1
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
		input_size=self.U.shape[1]
		hidden_size=self.W.shape[1]
		output_size=self.V.shape[0]
		return input_size,hidden_size,output_size

	def runTokens(self,states,ground_truth):
		num=len(states)
		err=0.0
		outputs=[]
		hidden_states=self.s

		for i in xrange(num):
			input_part=np.dot(self.U,states[i])
			recur_part=np.dot(self.W,hidden_states)
			hidden_states=gradient.sigmoid(input_part+recur_part)
			proj=np.dot(self.V,hidden_states)
			soft=softmax.softmax(proj)
			logsoft=np.log(soft)

			err=err+np.dot(ground_truth[i],logsoft)
			outputs.append(soft)

		err=err/num
		return err,outputs

	def update(self,learning_rates):
		if self.buffer>0:
			self.U=self.U-self.gU*learning_rates['U']/self.buffer
			self.W=self.W-self.gW*learning_rates['W']/self.buffer
			self.V=self.V-self.gV*learning_rates['V']/self.buffer
			self.s=self.s-self.gs*learning_rates['s']/self.buffer

			self.buffer=0
			self.gU=np.zeros(self.gU.shape)
			self.gW=np.zeros(self.gW.shape)
			self.gV=np.zeros(self.gV.shape)
			self.gs=np.zeros(self.gs.shape)


