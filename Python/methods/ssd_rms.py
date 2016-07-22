import sys
sys.path.insert(0,'../models/')
sys.path.insert(0,'../util/')
import numpy as np

import gradient
import vectorNorm
import sharp
import batchProduct
import softmax
import RNN
from elementwise import *

'''
>>> RMSprop Optimizer based on SSD
>>> model: RNN
>>> states: Input states. 2-D array of shape S*N where S is the sequence length and N is the input dimension
>>> ground_truth: Labels. 2-D array of shape S*H where S is the sequence length and H is the number of labels
>>> alpha: float in (0,1). E_t=alpha*E_{t-1}+(1-alpha)*g_t^2 para-=\\frac{\\eta}{\sqrt{E_t+\\epsilon}}
>>> trainonly: Controling training or test model. Boolean.
'''
def rms(model,states,ground_truth,alpha=0.5,trainonly=True):

		if trainonly:

			input_size,hidden_size,output_size=model.size()
			num=len(states)

			err=0.0
			tmpGradU=np.zeros(model.U.shape)
			tmpGradW=np.zeros(model.W.shape)
			tmpGradV=np.zeros(model.V.shape)
			tmpGrads=np.zeros(model.s.shape)

			hidden_states=model.s
			dSdW=np.zeros([hidden_size,hidden_size,hidden_size])
			dSdU=np.zeros([hidden_size,input_size,hidden_size])
			dSds=np.eye(hidden_size)

			lambdaW=np.ones([hidden_size,hidden_size])
			lambdaU=np.ones([hidden_size,input_size])
			VW=np.zeros([hidden_size,hidden_size])
			VU=np.zeros([hidden_size,input_size])

	        for index in xrange(num):
				input_part=np.dot(model.U,states[index])
				recur_part=np.dot(model.W,hidden_states)
				new_states=gradient.sigmoid(input_part+recur_part)
				proj=np.dot(model.V,new_states)
				soft=softmax.softmax(proj)
				logsoft=np.log(soft)
				err-=np.dot(ground_truth[index],logsoft)

				dis=soft-ground_truth[index]
				lamb=gradient.dsigmoid(input_part+recur_part)
				Lamb=np.diag(lamb)

				dEdV=np.dot(dis.reshape(output_size,1),new_states.reshape(1,hidden_size))
				tmpGradV+=dEdV

				dSdW=batchProduct.nXone(dSdW,model.W.transpose())
				dSdU=batchProduct.nXone(dSdU,model.W.transpose())
				for i in xrange(hidden_size):
					for j in xrange(hidden_size):
						dSdW[i,j,i]+=hidden_states[j]
					for j in xrange(input_size):
						dSdU[i,j,i]+=states[index][j]

				dSdW=batchProduct.nXone(dSdW,Lamb)
				dSdU=batchProduct.nXone(dSdU,Lamb) 
				dEdS=np.dot(model.V.transpose(),dis.reshape(output_size,1))
				dEdW=batchProduct.nXone(dSdW,dEdS).squeeze()
				dEdU=batchProduct.nXone(dSdU,dEdS).squeeze()

				VW=alpha*VW+(1-alpha)*np.power(dEdW,2)
				VU=alpha*VU+(1-alpha)*np.power(dEdW,2)
				DW=np.sqrt(lambdaW+np.sqrt(VW))
				DU=np.sqrt(lambdaU+np.sqrt(VU))
				dEdW_sharp=np.divide(sharp.sharp(np.divide(dEdW,DW)), DW)
				dEdU_sharp=np.divide(sharp.sharp(np.divide(dEdU,DU)), DU)
				
				tmpGradW+=dEdW_sharp
				tmpGradU+=dEdU_sharp
				        
				dSds=np.dot(np.dot(Lamb,model.W),dSds)
				dEds=np.dot(dSds.transpose(),dEdS).squeeze()
				tmpGrads+=dEds

				hidden_states=new_states

			err=err/num
			model.buffer+=1
			model.gU+=tmpGradU/num
			model.gW+=tmpGradW/num
			model.gV+=tmpGradV/num
			model.gs+=tmpGrads/num

		else:

			test_input_size,test_hidden_size,test_output_size=model.size()
			test_num=len(states)

			test_err=0.0
			    
			test_hidden_states=model.s

			for index in xrange(test_num):
				test_input_part=np.dot(model.U,states[index])
				test_recur_part=np.dot(model.W,test_hidden_states)
				test_new_states=gradient.sigmoid(test_input_part+test_recur_part)
				test_proj=np.dot(model.V,test_new_states)
				test_soft=softmax.softmax(test_proj)
				test_logsoft=np.log(test_soft)
				test_err-=np.dot(ground_truth[index],test_logsoft)

			err=test_err/test_num

	return err
