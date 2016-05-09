require 'math'

require 'util.gradient'
require 'util.loader'
require 'util.matrixNorm'
require 'util.randSVD'
require 'util.sharp'
require 'util.vectorNorm'
require 'models.RNN'
require 'optimization.sgd'
require 'optimization.ssd'

--hyper-parameters
param={}
param.trainXFile='../atis/train_word200.csv'
param.trainYFile='../atis/train_label200.csv'
param.testXFile='../atis/test_word50.csv'
param.testYFile='../atis/test_label50.csv'
param.dictFile='../atis/dict10.csv'
param.vectorDim=10
param.window=2
param.hiddens=30
param.batch=10
param.epoch=5
param.sgdLearningRateUWV=5
param.sgdUWVDecay=1
param.sgdLearningRateS=5
param.sgdSDecay=1
param.ssdLearningRateUWV=20
param.ssdUWVDecay=1
param.ssdLearningRateS=5
param.ssdSDecay=1
param.inputs=param.window*param.vectorDim

trainIndex=loader:loadData(param.trainXFile)
trainLabel=loader:loadData(param.trainYFile)
trainNum=table.getn(trainIndex)
testIndex=loader:loadData(param.testXFile)
testLabel=loader:loadData(param.testYFile)
testNum=table.getn(testIndex)
dict=loader:loadDict(param.dictFile)
dict[-1]=torch.rand(param.vectorDim)*0.05		--padding vector
--count #vocabulary
wordIndex={}
wordCount=0
for i=1,trainNum do
	local sentenceLen=table.getn(trainIndex[i])
	for j=1,sentenceLen do
		local word=trainIndex[i][j]
		if dict[word]~=nil and wordIndex[word]==nil then
			wordCount=wordCount+1
			wordIndex[word]=wordCount
		end
	end
end
param.outputs=wordCount+1
print('outputs',param.outputs)


--create 2 RNNs that have the same parameters and share the intital states
rnn1={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,__init__=RNN.__init__,
run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn2={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,__init__=RNN.__init__,
run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn1:__init__(param.inputs,param.hiddens,param.outputs)
rnn2:__init__(param.inputs,param.hiddens,param.outputs)
rnn2.s:copy(rnn1.s)
rnn2.i2h.weight:copy(rnn1.i2h.weight)
rnn2.h2h.weight:copy(rnn1.h2h.weight)
rnn2.h2o.weight:copy(rnn1.h2o.weight)

--create the input and expected output of RNN
trainX={}
trainY={}
testX={}
testY={}
for i=1,trainNum do
	if i%100==0 then
		print(i,'/',trainNum)
	end

	local sentenceLen=table.getn(trainIndex[i])
	local singleInput=torch.zeros(sentenceLen-param.window,param.inputs)
	local singleLabel=torch.zeros(sentenceLen-param.window,param.outputs)
	for line=1,sentenceLen-param.window do
		for w=1,param.window do
			local word=trainIndex[i][line+w-1]
			if dict[word]==nil then
				dict[word]=torch.rand(param.vectorDim)*0.05
			end
			singleInput[{{line,line},{(w-1)*param.vectorDim+1,w*param.vectorDim}}]
				=dict[word]:reshape(1,param.vectorDim)
		end
		local word=trainIndex[i][line+param.window]
		if wordIndex[word]~=nil then
			singleLabel[line][wordIndex[word]]=1
		else
			singleLabel[line][param.outputs]=1
		end
	end	
	table.insert(trainX,singleInput)
	table.insert(trainY,singleLabel)
end
for i=1,testNum do
	if i%100==0 then
		print(i,'/',testNum)
	end

	local sentenceLen=table.getn(testIndex[i])
	local singleInput=torch.zeros(sentenceLen-param.window,param.inputs)
	local singleLabel=torch.zeros(sentenceLen-param.window,param.outputs)
	for line=1,sentenceLen-param.window do
		for w=1,param.window do
			local word=testIndex[i][line+w-1]
			if dict[word]==nil then
				dict[word]=torch.rand(param.vectorDim)*0.05
			end
			singleInput[{{line,line},{(w-1)*param.vectorDim+1,w*param.vectorDim}}]
				=dict[word]:reshape(1,param.vectorDim)
		end
		local word=testIndex[i][line+param.window]
		if wordIndex[word]~=nil then
			singleLabel[line][wordIndex[word]]=1
		else
			singleLabel[line][param.outputs]=1
		end
	end
	table.insert(testX,singleInput)
	table.insert(testY,singleLabel)
end

--SGD--
local sgd_lr_uwv=param.sgdLearningRateUWV
local sgd_lr_s=param.sgdLearningRateS
for iter=1,param.epoch do
	local err_total=0
	for i=1,trainNum do
		local len=table.getn(trainIndex[i])
		if len>0 then
			local err=sgd(rnn1,trainX[i],trainY[i])
			err_total=err_total+err
		end
		if i%param.batch==0 then
			rnn1:update(sgd_lr_uwv,sgd_lr_s)
			print('epoch=%d,index=%d/%d'%{iter,i,trainNum})
			print('accumulated error=%f'%err_total)
		end
	end
	rnn1:update(sgd_lr_uwv,sgd_lr_s)
	print('epoch %d completed'%{iter})
	print('total error=%f'%err_total)
	sgd_lr_uwv=sgd_lr_uwv*param.sgdUWVDecay
	sgd_lr_s=sgd_lr_s*param.sgdSDecay
end

--SSD--
local ssd_lr_uwv=param.ssdLearningRateUWV
local ssd_lr_s=param.ssdLearningRateS
for iter=1,param.epoch do
	local err_total=0
	for i=1,trainNum do
		local len=table.getn(trainIndex[i])
		if len>0 then
			local err=ssd(rnn2,trainX[i],trainY[i])
			err_total=err_total+err
		end
		if i%param.batch==0 then
			rnn2:update(ssd_lr_uwv,ssd_lr_s)
			print('epoch=%d,index=%d/%d'%{iter,i,trainNum})
			print('accumulated error=%f'%err_total)
		end
	end
	rnn2:update(ssd_lr_uwv,ssd_lr_s)
	print('epoch %d completed'%iter)
	print('total error=%f'%err_total)
	ssd_lr_uwv=ssd_lr_uwv*param.ssdUWVDecay
	ssd_lr_s=ssd_lr_s*param.ssdSDecay
end
