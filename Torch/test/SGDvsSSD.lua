require 'models.RNN'
require 'math'
require 'optimization.sgd'
require 'optimization.ssd'

rnn1={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,
__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,
updateUWV=RNN.updateUWV,updateS=RNN.updateS}
rnn2={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,
__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,
updateUWV=RNN.updateUWV,updateS=RNN.updateS}
N=50
H=50
K=50
S=10
iters=10

rnn1:__init__(N,H,K)
rnn2:__init__(N,H,K)
rnn2.i2h.weight:copy(rnn1.i2h.weight)
rnn2.h2h.weight:copy(rnn1.h2h.weight)
rnn2.h2o.weight:copy(rnn1.h2o.weight)
-- print('input',rnn.i2h.weight)
-- print('recur',rnn.h2h.weight)
-- print('output',rnn.h2o.weight)

states=torch.ones(S,N)
ground_truth=torch.zeros(S,K)
for i=1,S do
	spot=math.floor(math.random()*K)+1
	ground_truth[i][spot]=1
end

time=0
for i=1,iters do
	local begin=os.clock()
	local err=sgd(rnn1,states,ground_truth)
	rnn1:updateUWV(0.2)
	rnn1:updateS(0.2)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)

time=0
for i=1,iters do
	local begin=os.clock()
	local err=ssd(rnn2,states,ground_truth)
	rnn2:updateUWV(2)
	rnn2:updateS(0.2)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)