require 'math'

require 'util.gradient'
require 'util.loader'
require 'util.matrixNorm'
require 'util.randSVD'
require 'util.sharp'
require 'util.vectorNorm'
require 'models.RNN'
require 'sgd.sgd'
require 'ssd.ssd'

--hyper-parameters
param={}
param.trainXFile='./atis/train_word.csv'
param.trainYFile='./atis/train_label.csv'
param.testXFile='./atis/test_word.csv'
param.testYFile='./atis/test_label.csv'
param.dictFile='./atis/dict.csv'
param.vectorDim=300
param.window=3
param.hiddens=300
param.outputs=128
param.batch=25
param.epoch=10
param.sgdLearningRate=0.01
param.inputs=param.window*param.vectorDim
param.leftPad=math.floor(param.window/2)

trainIndex=loader:loadData(param.trainXFile)
trainLabel=loader:loadData(param.trainYFile)
testIndex=loader:loadData(param.testXFile)
testLabel=loader:loadData(param.testYFile)
dict=loader:loadDict(param.dictFile)
dict[-1]=torch.rand(param.vectorDim)*0.05		--padding vector

--create 2 RNNs that have the same parameters and share the intital states
rnn1={i2h=nil,h2h=nil,h2o=nil,buffer=0,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn2={i2h=nil,h2h=nil,h2o=nil,buffer=0,__init__=RNN.__init__,run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn1:__init__(param.inputs,param.hiddens,param.outputs)
rnn2:__init__(param.inputs,param.hiddens,param.outputs)
rnn2.i2h.weight:copy(rnn2.i2h.weight)
rnn2.h2h.weight:copy(rnn2.h2h.weight)
rnn2.h2o.weight:copy(rnn2.h2o.weight)
initial=torch.rand(param.hiddens)*0.05

--create the input and expected output of RNN
trainNum=table.getn(trainIndex)
testNum=table.getn(testIndex)
trainX={}
trainY={}
testX={}
testY={}
for i=1,trainNum do
	if i%100==0 then
		print(i,'/',trainNum)
	end
	local sentenceLen=table.getn(trainIndex[i])
	local singleInput=torch.zeros(sentenceLen,param.inputs)
	local singleLabel=torch.zeros(sentenceLen,param.outputs)
	for p=2,sentenceLen+param.window do
		local toFill=dict[-1]
		if p>param.leftPad+1 and p<=sentenceLen+param.leftPad+1 then
			local word=trainIndex[i][p-param.leftPad-1]
			if dict[word]==nil then
				dict[word]=torch.rand(param.vectorDim)*0.05
			end
			toFill=dict[word]
		end
		local startline=math.max(1,p-param.window)
		local endline=math.min(sentenceLen,p-1)
		for line=startline,endline do
			local col=p-line
			singleInput[{{line,line},{(col-1)*param.vectorDim+1,col*param.vectorDim}}]=toFill:reshape(1,param.vectorDim)
		end
	end
	for w=1,sentenceLen do
		singleLabel[w][trainLabel[i][w]+1]=1
	end
	table.insert(trainX,singleInput)
	table.insert(trainY,singleLabel)
end

for i=1,testNum do
	if i%100==0 then
		print(i,'/',testNum)
	end
	local sentenceLen=table.getn(testIndex[i])
	local singleInput=torch.zeros(sentenceLen,param.inputs)
	local singleLabel=torch.zeros(sentenceLen,param.outputs)
	for p=2,sentenceLen+param.window do
		local toFill=dict[-1]
		if p>param.leftPad+1 and p<=sentenceLen+param.leftPad+1 then
			local word=testIndex[i][p-param.leftPad-1]
			if dict[word]==nil then
				dict[word]=torch.rand(param.vectorDim)*0.05
			end
			toFill=dict[word]
		end
		local startline=math.max(1,p-param.window)
		local endline=math.min(sentenceLen,p-1)
		for line=startline,endline do
			local col=p-line
			singleInput[{{line,line},{(col-1)*param.vectorDim+1,col*param.vectorDim}}]=toFill:reshape(1,param.vectorDim)
		end
	end
	for w=1,sentenceLen do
		singleLabel[w][testLabel[i][w]+1]=1
	end
	table.insert(testX,singleInput)
	table.insert(testY,singleLabel)
end

--SGD--
for iter=1,param.epoch do
	local err_total=0
	for i=1,trainNum do
		local len=table.getn(trainIndex[i])
		local err=sgd(rnn1,initial,trainX[i],trainY[i],len)
		err_total=err_total+err
		if i%param.batch==0 then
			rnn1:update(param.sgdLearningRate)
		end
		print('epoch=%d,index=%d'%{iter,i})
	end
	rnn1:update(param.sgdLearningRate)
	print('epoch %d completed'%iter)
	print('total error=%f'%err_total)
end


