require 'util.loader'

trainData=loader:loadData('./atis/train_word.csv')	--very fast
print('train data loaded')
trainLabel=loader:loadData('./atis/train_label.csv')
print('train label loaded')
dict=loader:loadDict('./atis/dict.csv')	--a bit slow
print('dict loaded')

-- print(trainData)
-- print(dict)