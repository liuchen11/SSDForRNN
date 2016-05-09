import sys
sys.path.insert(0,'../models/')
sys.path.insert(0,'../util/')

import gradient
import vectorNorm
import matrixNorm
import sharp
import batchProduct
import RNN
import numpy as np

def ssd(model,states,ground_truth):
	input_size,hidden_size,output_size=model.size()
	num=len(states)

	tmpGradU=np.zeros(model.U.shape)
	tmpGradW=np.zeros(model.W.shape)
	tmpGradV=np.zeros(model.V.shape)
	err=0.0

	hidden_states=model.s
	dSdW=np.zeros([hidden_size,hidden_size,hidden_size])
	dSdU=np.zeros([hidden_size,input_size,hidden_size])
	dSds=np.eye(hidden_size)

	P_S1_vec=np.zeros(hidden_size)
	Q_S1_vec=np.zeros(hidden_size)
	W_infn=matrixNorm.norm(model.W,np.inf)
	max_Vp_2n=0
	for i in xrange(output_size):
		Vp_2n=vectorNorm.norm(model.V[i],2)
		if Vp_2n>max_Vp_2n:
			max_Vp_2n=Vp_2n

	for index in xrange(num):
		input_part=np.dot(model.U,states[index])
		recur_part=np.dot(model.W,hidden_states)
		new_states=gradient.sigmoid(input_part+recur_part)
		proj=np.dot(model.V,new_states)
		soft=softmax.softmax(proj)
		logsoft=np.log(soft)
		err-=np.dot(ground_truth[index],logsoft)

		dis=soft-ground_truth[index]
		lamb1=gradient.dsigmoid(input_part+recur_part)
		lamb2=gradient.ddsigmoid(input_part+recur_part)
		Lamb1=np.diag(lamb1)
		s_old_2n=vectorNorm.norm(hidden_states,2)
		s_new_2n=vectorNorm.norm(new_states,2)
		x_2n=vectorNorm.norm(states[index],2)

		dEdV=np.dot(dis.reshape(output_size,1),new_states.reshape(1,hidden_size))
		dEdV_sharp=sharp.sharp(dEdV)
		L_dEdV=s_new_2n**2/2
		tmpGradV+=dEdV_sharp/L_dEdV

		new_dSdW=batchProduct.nXone(dSdW,model.W.transpose())
		new_dSdU=batchProduct.nXone(dSdU,model.W.transpose())
		for i in xrange(hidden_size):
			for j in xrange(hidden_size):
				new_dSdW[i,j,i]+=hidden_states[j]
			for j in xrange(input_size):
				new_dSdU[i,j,i]+=states[index][j]
		new_dSdW=batchProduct.nXone(new_dSdW,Lamb1)
		new_dSdU=batchProduct.nXone(new_dSdU,Lamb1)

		dEdS=np.dot(model.V.transpose(),dis.reshape(output_size,1))
		dEdW=batchProduct.nXone(new_dSdW,dEdS).squeeze()
		dEdU=batchProduct.nXone(new_dSdU,dEdS).squeeze()

			


