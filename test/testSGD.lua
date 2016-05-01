require 'models.RNN'
require 'optimization.sgd'

rnn={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,
__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,
updateUWV=RNN.updateUWV,updateS=RNN.updateS}
N=5
H=5
K=5
S=50
iters=1000

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
	local err=sgd(rnn,states,ground_truth)
	if i%2==0 then
		rnn:updateUWV(0.1)
		rnn:updateS(0.1)
	end
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)