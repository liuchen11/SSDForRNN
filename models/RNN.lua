RNN={i2h=nil,h2h=nil,h2o=nil}

function RNN:__init__(input_size,hidden_size,output_size)
	--This is an implementaton of 1-layer RNN

	self.i2h=nn.Linear(input_size,hidden_size)		--input to hidden layer
	self.h2h=nn.Linear(hidden_size,hidden_size)		--recurrent connection
	self.h2o=nn.Linear(hidden_size,output_size)		--hidden to output layer
end

function RNN:run1Token(states)
	--1-layer RNN is 2-input 2-output neural network
	--RNN runs for only one iteration
	--states={input data,hidden states}

	local input_data=states[1]
	local past_state=states[2]

	local input_part=self.i2h(input_data)
	local recurrent_part=self.h2h(past_state)
	local present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})

	local proj=self.h2o(present_states)
	local logsoft=nn.LogSoftMax()(proj)

	return {present_states,logsoft}
end

function RNN:runTokens(states,ground_truth,num)
	--RNN runs for several iterations
	--num: the length of sequence
	--states: {sequence...,hidden states}
	--ground_truth: {expected outputs}

	local present_states=states[num+1]
	local outputs={}
	local err=0.0

	for iter=1,num do
		local input_part=self.i2h(states[iter])
		local recurrent_part=self.h2h(present_states)
		present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})

		local proj=self.h2o(present_states)
		local logsoft=nn.LogSoftMax()(proj)

		err=err+ground_truth[iter]*torch.log(logsoft)
		table.insert(outputs,logsoft)
	end

	--outputs={outputs,hidden_states}
	table.insert(outputs,present_states)
	err=err/num
	return {error=err,states=outputs}
end