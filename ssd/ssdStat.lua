require 'math'
require 'models.RNN'
require 'util.gradient'
require 'util.matrixNorm'
require 'util.vectorNorm'
require 'util.sharp'

--stochastic spectral descent
function ssdStat(model,initial,states,ground_truth,num)
	--model: RNN model
	--initial: 
	--states: {input data..., initial hidden states}
	--ground_truth: {expected outputs}
	--num: length of the sequence
	--learning_rate: learning rate

	local input_size=model.i2h.weight:size(2)
	local hidden_size=model.h2h.weight:size(2)
	local output_size=model.h2o.weight:size(1)

	local U=model.i2h.weight
	local W=model.h2h.weight
	local V=model.h2o.weight
	local tmpGradU=torch.zeros(hidden_size,input_size)
	local tmpGradW=torch.zeros(hidden_size,hidden_size)
	local tmpGradV=torch.zeros(output_size,hidden_size)
	local err=0.0

	local present_states=initial
	local dSdW=torch.zeros(hidden_size,hidden_size,hidden_size)		--dSdW[i,j,k]=dS(k)/dW(i,j)
	local dSdU=torch.zeros(hidden_size,input_size,hidden_size)		--dSdU[i,j,k]=dS(k)/dU(i,j)
	local batchWT=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},torch.LongStorage{0,hidden_size,1})
	batchWT[1]=W:t()

	local T_updateV=0
	local T_updateW=0
	local T_updateU=0
	local T_UV=0
	local T_basic=0
	for iter=1,num do
		local begin=0
		local finish=0
		--save the status of previous time
		local past_states=present_states
		local past_dSdW=dSdW
		local past_dSdU=dSdU

		local input_part=model.i2h(states[iter])
		local recurrent_part=model.h2h(present_states)
		present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})
		local proj=model.h2o(present_states)
		local logsoft=nn.LogSoftMax()(proj)
		err=err+torch.dot(ground_truth[iter],logsoft)

		--update V's gradient
		--calculate dE/dV
		begin=os.clock()
		local dV=(ground_truth[iter]-logsoft):reshape(output_size,1)*present_states:reshape(1,hidden_size)
		local dV_sharp=sharp:sharp(dV)
		--calculate L_dV
		local L_dV=math.pow(vectorNorm:norm(present_states,2),2)/2
		tmpGradV=tmpGradV+dV_sharp/L_dV
		finish=os.clock()
		T_updateV=T_updateV+finish-begin
		--update W's and U's gradient
		--calculate dE/dW dE/dU
		begin=os.clock()
		local gS=gradient:dsigmoid(input_part+recurrent_part)

		local batchgS=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},torch.LongStorage{0,hidden_size,1})
		batchgS[1]=torch.diag(gS)

		dSdW=torch.bmm(dSdW,batchWT)
		dSdU=torch.bmm(dSdU,batchWT)
		for i=1,hidden_size do
			for j=1,hidden_size do
				dSdW[i][j][i]=dSdW[i][j][i]+past_states[j]
			end
			for j=1,input_size do
				dSdU[i][j][i]=dSdU[i][j][i]+states[iter][j]
			end
		end
		dSdW=torch.bmm(dSdW,batchgS)
		dSdU=torch.bmm(dSdU,batchgS)

		local dS=V:t()*(ground_truth[iter]-logsoft):reshape(output_size,1)		--dE/dS
		local batchdS=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,1},torch.LongStorage{0,1,1})
		batchdS[1]=dS
		local dW=torch.bmm(dSdW,batchdS):squeeze()
		local dU=torch.bmm(dSdU,batchdS):squeeze()
		local dW_sharp=sharp:sharp(dW)
		local dU_sharp=sharp:sharp(dU)
		finish=os.clock()
		T_UV=T_UV+finish-begin
		--calculate basic parameters
		begin=os.clock()
		local past_states_2n=vectorNorm:norm(past_states,2) --|s_{t-1}|_2
		local input_2n=vectorNorm:norm(states[iter],2)		--|x_t|_2
		local W_sinf=matrixNorm:norm(W,1/0,100)					--|W|_S^\inf
		local p=-1 											--find max_p |V_{p,:}|
		local max_Vp_2n=0									--!approximate the maximum of the output vector by choosen the row of V with largest 2-norm
		for i=1,output_size do
			local Vp_2n=vectorNorm:norm(V[i],2)
			if Vp_2n>max_Vp_2n then
				p=i
				max_Vp_2n=Vp_2n
			end
		end
		local max_Vp_infn=vectorNorm:norm(V[p],1/0)			--|V_{p,:}|_\inf
		local Vp=V[p]:reshape(hidden_size,1)
		local batchVp=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,1},torch.LongStorage{0,1,1})
		finish=os.clock()
		T_basic=T_basic+finish-begin
		--calculate L_dW
		begin=os.clock()
		local PW=torch.bmm(past_dSdW,batchWT)				--|ds/dW * W^T|
		local PWV=torch.bmm(PW,batchVp)						--|ds/dW * W^T * V[p]|
		local PWV_s1=matrixNorm:norm(PWV:squeeze(),1,-1)
		local Vs_2n=max_Vp_2n*past_states_2n
		local L_dW_p1=math.pow(gradient.max_dsigmoid*(Vs_2n+PWV_s1),2)
		local PW_s1_vec=torch.zeros(hidden_size)			--a vector whose i-th elements is the s1 norm of ds/dW[][][i]
		for i=1,hidden_size do
			PW_s1_vec[i]=matrixNorm:norm(PW[{{},{},{i,i}}]:squeeze(),1,-1)
		end
		local PW_s1_vec_2n=vectorNorm:norm(PW_s1_vec,2)
		local past_dSdW_bar=torch.zeros(hidden_size,hidden_size)	--a matrix whose i,j-th elements is 2-norm of ds/dW[i][j][]
		for i=1,hidden_size do
			for j=1,hidden_size do
				past_dSdW_bar[i][j]=vectorNorm:norm(past_dSdW[i][j],2)
			end
		end
		local past_dSdW_bar_s1=matrixNorm:norm(past_dSdW_bar,1,-1)
		local L_dW_p2=gradient.max_ddsigmoid*max_Vp_infn*past_states_2n*(past_states_2n+PW_s1_vec_2n)
		local L_dW_p3=gradient.max_dsigmoid*max_Vp_2n*PW_s1_vec_2n
		local L_dW_p4=gradient.max_ddsigmoid*max_Vp_infn*W_sinf*past_dSdW_bar_s1*(past_states_2n+PW_s1_vec_2n)
		local L_dW_p5=gradient.max_dsigmoid*max_Vp_2n*past_dSdW_bar_s1
		local L_dW=L_dW_p1+L_dW_p2+L_dW_p3+L_dW_p4+L_dW_p5
		tmpGradW=tmpGradW+dW_sharp/L_dW
		finish=os.clock()
		T_updateW=T_updateW+finish-begin

		begin=os.clock()
		local QW=torch.bmm(past_dSdU,batchWT)
		local QWV=torch.bmm(QW,batchVp)
		local QWV_s1=matrixNorm:norm(QWV:squeeze(),1,-1)
		local Vx_2n=max_Vp_2n*input_2n
		local L_dU_p1=math.pow(gradient.max_dsigmoid*(Vx_2n+QWV_s1),2)
		local QW_s1_vec=torch.zeros(hidden_size)
		for i=1,hidden_size do
			QW_s1_vec[i]=matrixNorm:norm(QW[{{},{},{i,i}}]:squeeze(),1,-1)
		end
		local QW_s1_vec_2n=vectorNorm:norm(QW_s1_vec,2)
		local past_dSdU_bar=torch.zeros(hidden_size,input_size)
		for i=1,hidden_size do
			for j=1,input_size do
				past_dSdU_bar[i][j]=vectorNorm:norm(past_dSdU[i][j],2)
			end
		end
		local past_dSdU_bar_s1=matrixNorm:norm(past_dSdU_bar,1,-1)
		local L_dU_p2=gradient.max_ddsigmoid*max_Vp_infn*input_2n*(input_2n+QW_s1_vec_2n)
		local L_dU_p3=gradient.max_ddsigmoid*max_Vp_infn*W_sinf*past_dSdU_bar_s1*(input_2n+QW_s1_vec_2n)
		local L_dU=L_dU_p1+L_dU_p2+L_dU_p3
		tmpGradU=tmpGradU+dU_sharp/L_dU
		finish=os.clock()
		T_updateU=T_updateU+finish-begin
	end

	err=err/num
	model.buffer=model.buffer+1
	model.i2h.gradWeight=model.i2h.gradWeight+tmpGradU/num
	model.h2h.gradWeight=model.h2h.gradWeight+tmpGradW/num
	model.h2o.gradWeight=model.h2o.gradWeight+tmpGradV/num

	return {T_basic=T_basic,T_UV=T_UV,T_updateU=T_updateU,T_updateW=T_updateW,T_updateV=T_updateV}
end