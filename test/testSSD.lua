require 'models.RNN'
require 'ssd.ssd'

rnn={i2h=nil,h2h=nil,h2o=nil,buffer=0,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
N=10
H=10
K=10
S=10
iters=10

rnn:__init__(N,H,K)
-- print('input',rnn.i2h.weight)
-- print('recur',rnn.h2h.weight)
-- print('output',rnn.h2o.weight)

initial=torch.zeros(H):fill(0.1)
states=torch.randn(S,N)
ground_truth=torch.randn(S,K)

time=0
for i=1,iters do
	local begin=os.clock()
	local err=ssd(rnn,initial,states,ground_truth,S)
	rnn:update(1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)