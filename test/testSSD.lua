require 'models.RNN'
require 'ssd.ssd'

rnn={i2h=nil,h2h=nil,h2o=nil,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens}
H=10

rnn:__init__(3,H,2)
print('input',rnn.i2h.weight)
print('recur',rnn.h2h.weight)
print('output',rnn.h2o.weight)

initial=torch.zeros(H):fill(0.1)
states=torch.Tensor{{1,1,1},{2,1,5},{2,3,4},{2,2,0},{1,3,4},{3,3,1},{4,0,0},{2,3,2}}:reshape(8,3)
ground_truth=torch.Tensor{{1,0},{1,0},{0,1},{1,0},{0,1},{1,0},{1,0},{0,1}}:reshape(8,2)

time=0
for i=1,10 do
	local begin=os.clock()
	local err=ssd(rnn,initial,states,ground_truth,7,1)
	local finish=os.clock()
	time=time+finish-begin
	print(i,err)
end
print('time',time)