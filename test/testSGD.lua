require 'models.RNN'
require 'sgd.sgd'

rnn={i2h=nil,h2h=nil,h2o=nil,buffer=0,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
N=20
H=20
K=20
S=10
iters=30

rnn:__init__(N,H,K)
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
	local err=sgd(rnn,initial,states,ground_truth,S)
	rnn:update(0.1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)