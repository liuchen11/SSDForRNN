require 'math'
require 'nn'
require 'models.RNN'
require 'util.gradient'
require 'util.matrixNorm'
require 'util.vectorNorm'
require 'util.sharp'

--stochastic spectral descent
function ssd(model,states,ground_truth)
	--model: RNN model
	--states: {sequence of input data} : 2-d matrix
	--ground_truth: {expected outputs} : 2-d matrix

	local input_size=model.i2h.weight:size(2)
	local hidden_size=model.h2h.weight:size(2)
	local output_size=model.h2o.weight:size(1)
	local num=states:size(1)

	local U=model.i2h.weight
	local W=model.h2h.weight
	local V=model.h2o.weight
	local tmpGradU=torch.zeros(hidden_size,input_size)
	local tmpGradW=torch.zeros(hidden_size,hidden_size)
	local tmpGradV=torch.zeros(output_size,hidden_size)
	local tmpGradS0=torch.zeros(hidden_size)
	local err=0.0

	local hidden_states=model.s
	local dSdW=torch.zeros(hidden_size,hidden_size,hidden_size)		--dSdW[i,j,k]=dS[k]/dW[i,j]
	local dSdU=torch.zeros(hidden_size,input_size,hidden_size)		--dSdU[i,j,k]=dS[k]/dU[i,j]
	local dSdS0=torch.eye(hidden_size)								--dSdS0[i,j]=dS[i]/dS0[j]
	--calculate some basic parameters
	local batchWT=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},
		torch.LongStorage{0,hidden_size,1})
	batchWT[1]=W:t()
	local P_S1_vec=torch.zeros(hidden_size)			--\|P_{t-1}\|_{S^1}
	local Q_S1_vec=torch.zeros(hidden_size)			--\|Q_{t-1}\|_{S^1}
	local max_Vp_2n=0								--\max_p \|V_p\|_2
	local W_infn=matrixNorm:norm(W,1/0,-1)			--\|W\|_{S^\infty}
	for p=1,output_size do
		local len=vectorNorm:norm(V[p],2)
		if max_Vp_2n<len then
			max_Vp_2n=len
		end
	end

	for iter=1,num do
		local input_part=model.i2h(states[iter])
		local recur_part=model.h2h(hidden_states)
		local new_state=nn.Sigmoid()(nn.CAddTable(){input_part,recur_part})
		local proj=model.h2o(new_state)
		local logsoft=nn.LogSoftMax()(proj)
		err=err-torch.dot(ground_truth[iter],logsoft)

		--calculate some basic medium results to calculate the gradient
		local softmax=logsoft:apply(function(x) return math.exp(x) end)
		local dis=softmax-ground_truth[iter]
		local lambda1=gradient:dsigmoid(input_part+recur_part)		--\dsigmoid(Ux_t+Ws_{t-1})
		local Lambda1=torch.diag(lambda1)
		local batchLambda1=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},
			torch.LongStorage{0,hidden_size,1})
		batchLambda1[1]=Lambda1
		local lambda1_infn=vectorNorm:norm(lambda1,1/0)
		local lambda2=gradient:ddsigmoid(input_part+recur_part)		--\ddsigmoid(Ux_t+Ws_{t-1})
		local s_old_2n=vectorNorm:norm(hidden_states,2)				--\|s_{t-1}\|_2
		local s_new_2n=vectorNorm:norm(new_state,2)				--\|s_t\|_2
		local x_2n=vectorNorm:norm(states[iter],2)

		--update V's gradient
		local dEdV=dis:reshape(output_size,1)*new_state:reshape(1,hidden_size)
		local dEdV_sharp=sharp:sharp(dEdV)
		local L_dEdV=math.pow(s_new_2n,2)/2
		tmpGradV=tmpGradV+dEdV_sharp/L_dEdV

		--update W's and U's gradient
		----calculate dS_t/dW, dS_t/dU
		local new_dSdW=torch.bmm(dSdW,batchWT)
		local new_dSdU=torch.bmm(dSdU,batchWT)
		for i=1,hidden_size do
			for j=1,hidden_size do
				new_dSdW[i][j][i]=new_dSdW[i][j][i]+hidden_states[j]
			end
			for j=1,input_size do
				new_dSdU[i][j][i]=new_dSdU[i][j][i]+states[iter][j]
			end
		end
		new_dSdW=torch.bmm(new_dSdW,batchLambda1)
		new_dSdU=torch.bmm(new_dSdU,batchLambda1)
		----calculate dE/dS_t
		local dEdS=V:t()*dis:reshape(output_size,1)
		local batchdEdS=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,1},
			torch.LongStorage{0,1,1})
		batchdEdS[1]=dEdS
		local dEdW=torch.bmm(new_dSdW,batchdEdS):squeeze()
		local dEdU=torch.bmm(new_dSdU,batchdEdS):squeeze()
		----calculate L_dW, L_dU
		------calculate \|P_t_{:,:,p}\| and \|Q_t_{:,:,p}\|
		local new_dSdW_perm=new_dSdW:permute(3,1,2)
		local new_dSdU_perm=new_dSdU:permute(3,1,2)
		local new_P_S1_vec=torch.zeros(hidden_size)			-- \|P_t_{:,:,p}\|_{S^1}
		local new_Q_S1_vec=torch.zeros(hidden_size)			-- \|Q_t_{:,:,p}\|_{S^1}
		for i=1,hidden_size do
			new_P_S1_vec[i]=matrixNorm:norm(new_dSdW_perm[i],1,-1)
			new_Q_S1_vec[i]=matrixNorm:norm(new_dSdU_perm[i],1,-1)
		end
		------calculate \max_p \lambda'_p^{-1}\lambda''_p \|P_{t-1_{:,:,p}}\|_{S^1}
			--and \max_p \lambda'_p^{-1}\lambda''_p \|Q_{t-1_{:,:,p}}\|_{S^1}
		--update all parameters
		local max_lambda_P_S1=0
		local max_lambda_Q_S1=0
		for i=1,hidden_size do
			local lambda_P_S1=P_S1_vec[i]*lambda2[i]/lambda1[i]
			local lambda_Q_S1=Q_S1_vec[i]*lambda2[i]/lambda1[i]
			if lambda_P_S1>max_lambda_P_S1 then
				max_lambda_P_S1=lambda_P_S1
			end
			if lambda_Q_S1>max_lambda_Q_S1 then
				max_lambda_Q_S1=lambda_Q_S1
			end
		end
		------calulate L_dEdW
		local new_P_S1_vec_2n=vectorNorm:norm(new_P_S1_vec,2)
		local P_S1_vec_2n=vectorNorm:norm(P_S1_vec,2)
		local L_dEdW1=math.pow(new_P_S1_vec_2n*max_Vp_2n,2)/2
		local L_dEdW2=max_Vp_2n*s_old_2n*max_lambda_P_S1
		local L_dEdW3=2*lambda1_infn*max_Vp_2n*P_S1_vec_2n
		local L_dEdW4=W_infn*max_Vp_2n*max_lambda_P_S1*P_S1_vec_2n
		local L_dEdW=L_dEdW1+L_dEdW2+L_dEdW3+L_dEdW4
		local dEdW_sharp=sharp:sharp(dEdW)
		tmpGradW=tmpGradW+dEdW_sharp/L_dEdW
		------calulate L_dEdU
		local new_Q_S1_vec_2n=vectorNorm:norm(new_Q_S1_vec,2)
		local Q_S1_vec_2n=vectorNorm:norm(Q_S1_vec,2)
		local L_dEdU1=math.pow(new_Q_S1_vec_2n*max_Vp_2n,2)/2
		local L_dEdU2=max_Vp_2n*x_2n*max_lambda_Q_S1
		local L_dEdU3=max_Vp_2n*W_infn*Q_S1_vec_2n*max_lambda_Q_S1
		local L_dEdU=L_dEdU1+L_dEdU2+L_dEdU3
		local dEdU_sharp=sharp:sharp(dEdU)
		tmpGradU=tmpGradU+dEdU_sharp/L_dEdU

		--update s0's gradient
		dSdS0=Lambda1*W*dSdS0
		local dEdS0=dSdS0:t()*dEdS
		tmpGradS0=tmpGradS0+dEdS0

		--update parameters
		hidden_states=new_state
		dSdW=new_dSdW
		dSdU=new_dSdU
		P_S1_vec=new_P_S1_vec
		Q_S1_vec=new_Q_S1_vec
	end

	--update the average gradients
	err=err/num
	model.buffer=model.buffer+1
	model.i2h.gradWeight=model.i2h.gradWeight+tmpGradU/num
	model.h2h.gradWeight=model.h2h.gradWeight+tmpGradW/num
	model.h2o.gradWeight=model.h2o.gradWeight+tmpGradV/num
	model.ds=model.ds+tmpGradS0/num

	--return the error using previous parameters
	return err
end