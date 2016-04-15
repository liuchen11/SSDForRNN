require 'models.RNN'
require 'ssd.ssdStat'

rnn={i2h=nil,h2h=nil,h2o=nil,buffer=0,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
N=30
H=30
K=30
S=30
iters=10

rnn:__init__(N,H,K)
-- print('input',rnn.i2h.weight)
-- print('recur',rnn.h2h.weight)
-- print('output',rnn.h2o.weight)

initial=torch.randn(H)
states=torch.randn(S,N)
ground_truth=torch.zeros(S,K)

time=0
T_updateV=0
T_updateW=0
T_updateU=0
T_UV=0
T_basic=0

for i=1,iters do
	local begin=os.clock()
	local ret=ssdStat(rnn,initial,states,ground_truth,S)
	rnn:update(1)
	local finish=os.clock()
	time=time+finish-begin
	T_updateU=T_updateU+ret.T_updateU
	T_updateW=T_updateW+ret.T_updateW
	T_updateV=T_updateV+ret.T_updateV
	T_UV=T_UV+ret.T_UV
	T_basic=T_basic+ret.T_basic
end
print('time',time)
print(string.format('T_upV\t%.4f\t%.4f',T_updateV,T_updateV/time*100))
print(string.format('T_upW\t%.4f\t%.4f',T_updateW,T_updateW/time*100))
print(string.format('T_upU\t%.4f\t%.4f',T_updateU,T_updateU/time*100))
print(string.format('T_UV\t%.4f\t%.4f',T_UV,T_UV/time*100))
print(string.format('T_basic\t%.4f\t%.4f',T_basic,T_basic/time*100))