require 'models.RNN'

rnn={i2h=nil,h2h=nil,h2o=nil,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens}
rnn:__init__(2,2,2)
print('input',rnn.i2h.weight)
print('recur',rnn.h2h.weight)
print('output',rnn.h2o.weight)

states=torch.Tensor{{1,1},{1,3},{2,2},{3,2},{1,3},{2,3},{0,1},{0,0}}:reshape(8,2)
ground_truth=torch.Tensor{{1,0},{1,0},{0,1},{0,1},{0,1},{1,0},{1,0}}:reshape(7,2)
ret=rnn:runTokens(states,ground_truth,7)
print('U',rnn.i2h.weight)
print('W',rnn.h2h.weight)
print('V',rnn.h2o.weight)
print('error',ret.error)
print('outputs')
for i=1,7 do
	print(ret.states[i])
end