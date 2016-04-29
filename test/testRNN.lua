require 'models.RNN'

rnn={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,
__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,
updateUWV=RNN.updateUWV,updateS=RNN.updateS}

rnn:__init__(2,3,2)
print('input',rnn.i2h.weight)
print('recur',rnn.h2h.weight)
print('output',rnn.h2o.weight)
print('hidden_states',rnn.s)

states=torch.Tensor{{1,1},{1,3},{2,2},{3,2},{1,3},{2,3},{0,1}}:reshape(7,2)
ground_truth=torch.Tensor{{1,0},{1,0},{0,1},{0,1},{0,1},{1,0},{1,0}}:reshape(7,2)
ret=rnn:run1Token(states[1])
ret=rnn:runTokens(states,ground_truth)
print('U',rnn.i2h.weight)
print('W',rnn.h2h.weight)
print('V',rnn.h2o.weight)
print('error',ret.error)
print('outputs')
for i=1,7 do
	print(ret.prediction[i])
end