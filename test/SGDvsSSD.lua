require 'models.RNN'
require 'math'
require 'sgd.sgd'
require 'ssd.ssd'

rnn1={i2h=nil,h2h=nil,h2o=nil,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens}
rnn2={i2h=nil,h2h=nil,h2o=nil,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens}
N=20
H=10
K=20
S=100
iters=10

rnn1:__init__(N,H,K)
rnn2:__init__(N,H,K)
rnn2.i2h.weight:copy(rnn2.i2h.weight)
rnn2.h2h.weight:copy(rnn2.h2h.weight)
rnn2.h2o.weight:copy(rnn2.h2o.weight)
-- print('input',rnn.i2h.weight)
-- print('recur',rnn.h2h.weight)
-- print('output',rnn.h2o.weight)

initial=torch.randn(H)
states=torch.randn(S,N)
ground_truth=torch.zeros(S,K)
for i=1,S do
	spot=math.floor(math.random()*K)+1
	ground_truth[i][spot]=1
end

time=0
for i=1,iters do
	local begin=os.clock()
	local err=sgd(rnn1,initial,states,ground_truth,S,1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)

time=0
for i=1,iters do
	local begin=os.clock()
	local err=ssd(rnn2,initial,states,ground_truth,S,1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)