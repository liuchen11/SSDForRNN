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
require 'optimization.ssd_simp'

--hyper-parameters
param={}
param.trainXFile='../atis/train_word1000.csv'
param.trainYFile='../atis/train_label1000.csv'
param.testXFile='../atis/test_word50.csv'
param.testYFile='../atis/test_label50.csv'
param.dictFile='../atis/dict10.csv'
param.vectorDim=10
param.window=1
param.hiddens=10
param.outputs=128
param.batch=50
param.epoch=3
param.sgdLearningRateUWV=5
param.sgdUWVDecay=1
param.sgdLearningRateS=2
param.sgdSDecay=1
param.ssdLearningRateUWV=5
param.ssdUWVDecay=1
param.ssdLearningRateS=2
param.ssdSDecay=1
param.inputs=param.window*param.vectorDim
param.leftPad=math.floor(param.window/2)

trainIndex=loader:loadData(param.trainXFile)
trainLabel=loader:loadData(param.trainYFile)
testIndex=loader:loadData(param.testXFile)
testLabel=loader:loadData(param.testYFile)
dict=loader:loadDict(param.dictFile)
dict[-1]=torch.rand(param.vectorDim)*0.05		--padding vector

--create 2 RNNs that have the same parameters and share the intital states
rnn1={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,__init__=RNN.__init__,
run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn2={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,__init__=RNN.__init__,
run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn3={i2h=nil,h2h=nil,h2o=nil,s=nil,ds=nil,buffer=0,__init__=RNN.__init__,
run1Token=RNN.run1Token,runTokens=RNN.runTokens,update=RNN.update}
rnn1:__init__(param.inputs,param.hiddens,param.outputs)
rnn2:__init__(param.inputs,param.hiddens,param.outputs)
rnn3:__init__(param.inputs,param.hiddens,param.outputs)
rnn2.s:copy(rnn1.s)
rnn2.i2h.weight:copy(rnn1.i2h.weight)
rnn2.h2h.weight:copy(rnn1.h2h.weight)
rnn2.h2o.weight:copy(rnn1.h2o.weight)
rnn3.s:copy(rnn1.s)
rnn3.i2h.weight:copy(rnn1.i2h.weight)
rnn3.h2h.weight:copy(rnn1.h2h.weight)
rnn3.h2o.weight:copy(rnn1.h2o.weight)

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
			-- print(vectorNorm:norm(rnn1.i2h.gradWeight,2)/rnn1.i2h.gradWeight:nElement())
			-- print(vectorNorm:norm(rnn1.h2h.gradWeight,2)/rnn1.h2h.gradWeight:nElement())
			-- print(vectorNorm:norm(rnn1.h2o.gradWeight,2)/rnn1.h2o.gradWeight:nElement())
			-- print(vectorNorm:norm(rnn1.ds,2)/rnn1.ds:nElement())
			-- print(vectorNorm:norm(rnn1.s,2)/rnn1.s:nElement())
			-- print('------------------------------')
			-- print(vectorNorm:norm(rnn1.h2o.weight,2)/rnn1.h2o.weight:nElement())
			rnn1:update(sgd_lr_uwv,sgd_lr_s)
			print('epoch=%d,index=%d/%d'%{iter,i,trainNum})
			print('accumulated error=%f'%err_total)
			local u1,s1,v1=torch.svd(rnn1.i2h.weight)
			local u2,s2,v2=torch.svd(rnn1.h2h.weight)
			local u3,s3,v3=torch.svd(rnn1.h2o.weight)
			-- print(s1:reshape(1,s1:size(1)))
			-- print(s2:reshape(1,s2:size(1)))
			-- print(s3:reshape(1,s3:size(1)))
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
-- local ssd_lr_s=param.ssdLearningRateS
-- for iter=1,param.epoch do
-- 	local err_total=0
-- 	for i=1,trainNum do
-- 		local len=table.getn(trainIndex[i])
-- 		if len>0 then
-- 			local err=ssd(rnn2,trainX[i],trainY[i])
-- 			err_total=err_total+err
-- 		end
-- 		-- print(vectorNorm:norm(rnn2.i2h.gradWeight,2))
-- 		-- print(vectorNorm:norm(rnn2.h2h.gradWeight,2))
-- 		-- print(vectorNorm:norm(rnn2.h2o.gradWeight,2))
-- 		-- print(vectorNorm:norm(rnn2.ds,2))
-- 		if i%param.batch==0 then
-- 			rnn2:update(ssd_lr_uwv,ssd_lr_s)
-- 			print('epoch=%d,index=%d/%d'%{iter,i,trainNum})
-- 			print('accumulated error=%f'%err_total)
-- 		end
-- 	end
-- 	rnn2:update(ssd_lr_uwv,ssd_lr_s)
-- 	print('epoch %d completed'%iter)
-- 	print('total error=%f'%err_total)
-- 	ssd_lr_uwv=ssd_lr_uwv*param.ssdUWVDecay
-- 	ssd_lr_s=ssd_lr_s*param.ssdSDecay
-- end

--SSD_Simp
local ssd_simp_lr_uwv=5
local ssd_simp_lr_s=2
for iter=1,param.epoch do
	local err_total=0
	for i=1,trainNum do
		local len=table.getn(trainIndex[i])
		if len>0 then
			local err=ssd_simp(rnn3,trainX[i],trainY[i])
			err_total=err_total+err
		end
		-- print(vectorNorm:norm(rnn2.i2h.gradWeight,2))
		-- print(vectorNorm:norm(rnn2.h2h.gradWeight,2))
		-- print(vectorNorm:norm(rnn2.h2o.gradWeight,2))
		-- print(vectorNorm:norm(rnn2.ds,2))
		if i%param.batch==0 then
			rnn2:update(ssd_simp_lr_uwv,ssd_simp_lr_s)
			print('epoch=%d,index=%d/%d'%{iter,i,trainNum})
			print('accumulated error=%f'%err_total)
		end
	end
	rnn2:update(ssd_simp_lr_uwv,ssd_simp_lr_s)
	print('epoch %d completed'%iter)
	print('total error=%f'%err_total)
	ssd_simp_lr_uwv=ssd_simp_lr_uwv*1
	ssd_simp_lr_s=ssd_simp_lr_s*1
end
