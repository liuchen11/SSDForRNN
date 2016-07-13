import sys
sys.path.insert(0,'../models/')
sys.path.insert(0,'../util/')

import gradient
import vectorNorm
import matrixNorm
import sharp
import batchProduct
from elementwise import *
import softmax
import RNN
import numpy as np

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
		        #dEdV_sharp=sharp.sharp(dEdV)
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

                        VW=alpha*VW + (1-alpha)*elmtwiseprod(dEdW,dEdW)
                        VU=alpha*VU + (1-alpha)*elmtwiseprod(dEdU,dEdU)
                        DW=elmtwisesqrt(lambdaW + elmtwisesqrt(VW))
                        DU=elmtwisesqrt(lambdaU + elmtwisesqrt(VU))
                        dEdW_sharp=elmtwisediv(sharp.sharp(elmtwisediv(dEdW,DW)), DW)
                        dEdU_sharp=elmtwisediv(sharp.sharp(elmtwisediv(dEdU,DU)), DU)
                        
		        #dEdW_sharp=sharp.sharp(dEdW)
		        #dEdU_sharp=sharp.sharp(dEdU)
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
