require 'models.RNN'
require 'optimization.ssd'

rnn={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,
__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,
updateUWV=RNN.updateUWV,updateS=RNN.updateS}
N=10
H=10
K=10
S=10
iters=10

rnn:__init__(N,H,K)
-- print('input',rnn.i2h.weight)
-- print('recur',rnn.h2h.weight)
-- print('output',rnn.h2o.weight)

states=torch.randn(S,N)
ground_truth=torch.zeros(S,K)
for i=1,S do
	spot=math.floor(math.random()*K)+1
	ground_truth[i][spot]=1
end

time=0
for i=1,iters do
	local begin=os.clock()
	local err=ssd(rnn,states,ground_truth)
	rnn:updateUWV(0.1)
	rnn:updateS(0.1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)