RNN=require 'models.RNN'
gradient=require 'util.gradient'

function sgd(model,states,ground_truth,num,learning_rate)
	--model: RNN model
	--states: {input data..., initial hidden states}
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

	local present_states=states[num+1]
	local last_dSdW={}		--ds(t)/dW each element should be a 3d tensor
	local last_dSdU={}		--ds(t)/dU each element should be a 3d tensor

	for iter=1,num do
		local input_part=model.i2h(states[iter])
		local recurrent_part=model.h2h(present_states)
		present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})
		local proj=model.h2o(present_states)
		local logsoft=nn.LogSoftMax()(proj)

		--update V's gradient
		local dV=(ground_truth[iter]-logsoft):reshape(output_size,1)*present_states:reshape(1,hidden_size)
		model.h2o.gradWeight=model.h2o.gradWeight+dV
		--update W's and U's gradient
		local dSdW=torch.zeros(hidden_size,hidden_size,hidden_size)		--dSdW[i,j,k]=dS(k)/dW(i,j)
		local dSdU=torch.zeros(hidden_size,input_size,hidden_size)		--dSdU[i,j,k]=dS(k)/dW(i,j)
		local gS=gradient:dsigmoid(nn.CAddTable(){input_part,recurrent_part})
		if iter==1 then
			for i=1,hidden_size do
				for j=1,hidden_size do
					dSdW[i][j][i]=gS[i]*present_states[j]
				end
				for j=1,input_size do
					dSdU[i][j][i]=gS[i]*states[iter][j]
				end
			end
		else
			for i=1,hidden_size do
				for j=1,hidden_size do
					for k=1,hidden_size do
						if i==k then
							dSdW[i][j][k]=gS[k]*(present_states[j]+last_dSdW[i][j]*W[k])
						else
							dSdW[i][j][k]=gS[k]*last_dSdW[i][j]*W[k]
						end
					end
				end
				for j=1,hidden_size do
					for k=1,hidden_size do
						if i==k then
							dSdU[i][j][k]=gS[k]*(states[iter][j]+last_dSdU[i][j]*W[k])
						else
							dSdU[i][j][k]=gS[k]*last_dSdU[i][j]*W[k]
						end
					end
				end
			end
		end
		local dS=V:t()*(ground_truth-logsoft):reshape(output_size,1)
		local dW=torch.zeros(hidden_size,hidden_size)
		local dU=torch.zeros(hidden_size,input_size)
		for i=1,hidden_size do
			for j=1,hidden_size do
				dW[i][j]=dSdW[i][j]*dS
			end
			for j=1,input_size do
				dU[i][j]=dSdU[i][j]*dS
			end
		end
		model.h2h.gradWeight=model.h2h.gradWeight+dW
		model.i2h.gradWeight=model.i2h.gradWeight+dU
	end

	model.i2h.gradWeight=model.i2h.gradWeight/num
	model.h2h.gradWeight=model.h2h.gradWeight/num
	model.h2o.gradWeight=model.h2o.gradWeight/num

	model.i2h:updateParameters(learning_rate)
	model.h2h:updateParameters(learning_rate)
	model.h2o:updateParameters(learning_rate)
end