require 'models.RNN'
require 'util.gradient'

--stochastic gradient descent
function sgd(model,initial,states,ground_truth,num,learning_rate)
	--model: RNN model
	--initial: initial hidden states
	--states: {sequence of inputs}
	--ground_truth: {expected outputs}
	--num: length of the sequence
	--learning_rate: learning rate

	local input_size=model.i2h.weight:size()[2]
	local hidden_size=model.h2h.weight:size()[2]
	local output_size=model.h2o.weight:size()[1]

	model.i2h.gradWeight:fill(0)
	model.h2h.gradWeight:fill(0)
	model.h2o.gradWeight:fill(0)
	local U=model.i2h.weight
	local W=model.h2h.weight
	local V=model.h2o.weight
	local err=0.0

	local present_states=initial
	local dSdW=torch.zeros(hidden_size,hidden_size,hidden_size)		--dSdW[i,j,k]=dS(k)/dW(i,j)
	local dSdU=torch.zeros(hidden_size,input_size,hidden_size)		--dSdU[i,j,k]=dS(k)/dU(i,j)
	local batchWT=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,hidden_size},torch.LongStorage{0,hidden_size,1})
	batchWT[1]=W:t()

	for iter=1,num do
		local past_states=present_states
		local input_part=model.i2h(states[iter])
		local recurrent_part=model.h2h(present_states)
		present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})
		local proj=model.h2o(present_states)
		local logsoft=nn.LogSoftMax()(proj)
		err=err+torch.dot(ground_truth[iter],logsoft)

		--update V's gradient
		local dV=(ground_truth[iter]-logsoft):reshape(output_size,1)*present_states:reshape(1,hidden_size)
		model.h2o.gradWeight=model.h2o.gradWeight+dV
		--update W's and U's gradient
		local gS=gradient:dsigmoid(nn.CAddTable(){input_part,recurrent_part})

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

		local dS=(V:t()*(ground_truth[iter]-logsoft)):reshape(hidden_size,1)	--dE/dS
		local batchdS=torch.Tensor(torch.LongStorage{hidden_size,hidden_size,1},torch.LongStorage{0,1,1})
		batchdS[1]=dS
		local dW=torch.bmm(dSdW,batchdS):squeeze()		--dE/dW=dE/dS dS/dW
		local dU=torch.bmm(dSdU,batchdS):squeeze()		--dE/dU=dE/dS dS/dU

		model.h2h.gradWeight=model.h2h.gradWeight+dW
		model.i2h.gradWeight=model.i2h.gradWeight+dU
	end

	err=err/num
	model.i2h.gradWeight=model.i2h.gradWeight/num
	model.h2h.gradWeight=model.h2h.gradWeight/num
	model.h2o.gradWeight=model.h2o.gradWeight/num

	model.i2h:updateParameters(learning_rate)
	model.h2h:updateParameters(learning_rate)
	model.h2o:updateParameters(learning_rate)

	--return the error using previous parameters
	return err
end