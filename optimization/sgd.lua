require 'math'
require 'nn'
require 'models.RNN'
require 'util.gradient'

--stochastic gradient descent
function sgd(model,states,ground_truth)
	--model: RNN model
	--states: {sequence of inputs} : 2-d matrix
	--ground_truth: {expected outputs} : 2-d matrix

	local input_size=model.i2h.weight:size(2)
	local hidden_size=model.h2h.weight:size(2)
	local output_size=model.h2o.weight:size(1)
	local num=states:size(1)	

	local U=model.i2h.weight
	local W=model.h2h.weight
	local V=model.h2o.weight
	local s0=model.s
	local tmpGradU=torch.zeros(hidden_size,input_size)
	local tmpGradW=torch.zeros(hidden_size,hidden_size)
	local tmpGradV=torch.zeros(output_size,hidden_size)
	local tmpGradS0=torch.zeros(hidden_size)
	local err=0.0

	local hidden_states=model.s
	local dSdW=torch.zeros(hidden_size,hidden_size,hidden_size)			--dSdW[i,j,k]=dS[k]/dW[i,j]
	local dSdU=torch.zeros(hidden_size,input_size,hidden_size)			--dSdU[i,j,k]=dS[k]/dU[i,j]
	local dSdS0=torch.eye(hidden_size)
	local batchWT=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},
		torch.LongStorage{0,hidden_size,1})
	batchWT[1]=W:t()

	for iter=1,num do
		--compute the output and new hidden state
		local input_part=model.i2h(states[iter])
		local recur_part=model.h2h(hidden_states)
		local new_state=nn.Sigmoid()(nn.CAddTable(){input_part,recur_part})
		local proj=model.h2o(new_state)
		local logsoft=nn.LogSoftMax()(proj)
		err=err-torch.dot(ground_truth[iter],logsoft)

		--basic medium results used for updating parameters
		local softmax=logsoft:apply(function(x) return math.exp(x) end)
		local dis=softmax-ground_truth[iter]
		
		--update V's gradient
		local dEdV=dis:reshape(output_size,1)*new_state:reshape(1,hidden_size)
		tmpGradV=tmpGradV+dEdV
		
		--update W's and U's gradient
		local lambda=gradient:dsigmoid(input_part+recur_part)
		local Lambda=torch.diag(lambda)
		local batchLambda=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},
			torch.LongStorage{0,hidden_size,1})
		batchLambda[1]=Lambda
		--update dS/dW and dS/dU
		dSdW=torch.bmm(dSdW,batchWT)
		dSdU=torch.bmm(dSdU,batchWT)
		for i=1,hidden_size do
			for j=1,hidden_size do
				dSdW[i][j][i]=dSdW[i][j][i]+hidden_states[j]
			end
			for j=1,input_size do
				dSdU[i][j][i]=dSdU[i][j][i]+states[iter][j]
			end
		end
		dSdW=torch.bmm(dSdW,batchLambda)
		dSdU=torch.bmm(dSdU,batchLambda)
		--update dE/dS
		local dEdS=V:t()*dis:reshape(output_size,1)
		local batchdEdS=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,1},
			torch.LongStorage{0,1,1})
		batchdEdS[1]=dEdS
		local dEdW=torch.bmm(dSdW,batchdEdS):squeeze()
		local dEdU=torch.bmm(dSdU,batchdEdS):squeeze()
		--update tmpGrad
		tmpGradW=tmpGradW+dEdW
		tmpGradU=tmpGradU+dEdU

		--update s0's gradient
		dSdS0=Lambda*W*dSdS0
		local dEdS0=dSdS0:t()*dEdS
		tmpGradS0=tmpGradS0+dEdS0

		--update the hidden state
		hidden_states=new_state
	end

	err=err/num
	model.buffer=model.buffer+1
	model.i2h.gradWeight=model.i2h.gradWeight+tmpGradU/num
	model.h2h.gradWeight=model.h2h.gradWeight+tmpGradW/num
	model.h2o.gradWeight=model.h2o.gradWeight+tmpGradV/num
	model.ds=model.ds+tmpGradS0/num

	--return the error using previous parameters
	return err
end

