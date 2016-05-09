require 'nn'

RNN={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0}

function RNN:__init__(input_size,hidden_size,output_size)
	--This is an implementaton of 1-layer RNN

	--We disable bias of all layers
	self.i2h=nn.Linear(input_size,hidden_size,false)		--input to hidden layer
	self.h2h=nn.Linear(hidden_size,hidden_size,false)		--recurrent connection
	self.h2o=nn.Linear(hidden_size,output_size,false)		--hidden to output layer
	self.s=torch.randn(hidden_size)							--initial hidden value is initialized randomly
	self.i2h.gradWeight:fill(0)								--all derivatives are set zeros
	self.h2h.gradWeight:fill(0)
	self.h2o.gradWeight:fill(0)
	self.ds=torch.zeros(hidden_size)
	buffer=0
end

function RNN:run1Token(state)
	--RNN runs for only one iteration
	--states=one input

	local input_data=state
	local past_state=self.s

	local input_part=self.i2h(input_data)
	local recurrent_part=self.h2h(past_state)
	local present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})

	local proj=self.h2o(present_states)
	local logsoft=nn.LogSoftMax()(proj)

	return {present_states,logsoft}
end

function RNN:runTokens(states,ground_truth)
	--RNN runs for several iterations
	--initial: initial states
	--num: the length of sequence
	--states: {sequence of inputs}
	--ground_truth: {expected outputs}

	local present_states=self.s
	local outputs={}
	local num=states:size(1)
	local err=0.0

	for iter=1,num do
		local input_part=self.i2h(states[iter])
		local recurrent_part=self.h2h(present_states)
		present_states=nn.Sigmoid()(nn.CAddTable(){input_part,recurrent_part})

		local proj=self.h2o(present_states)
		local logsoft=nn.LogSoftMax()(proj)

		err=err+torch.dot(ground_truth[iter],logsoft)
		table.insert(outputs,logsoft)
	end

	err=err/num
	return {error=err,prediction=outputs}
end

function RNN:updateUWV(learning_rate)
	--update parameters U W V
	--learning_rate: learning rate
	if self.buffer>0 then
		self.i2h:updateParameters(learning_rate/self.buffer)
		self.h2h:updateParameters(learning_rate/self.buffer)
		self.h2o:updateParameters(learning_rate/self.buffer)

		self.buffer=0
		self.i2h.gradWeight:fill(0)
		self.h2h.gradWeight:fill(0)
		self.h2o.gradWeight:fill(0)
	end
end

function RNN:updateS(learning_rate)
	--update parameters s0
	--learning_rate: learning rate
	if self.buffer>0 then
		self.s=self.s-learning_rate*self.ds
		self.buffer=0
		self.ds:fill(0)
	end
end