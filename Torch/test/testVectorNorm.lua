require 'util.vectorNorm'
require 'math'

vectors=torch.randn(10000,10000)
results=torch.zeros(vectors:size(1))
begin=os.clock()
for i=1,vectors:size(1) do
	results[i]=vectorNorm:norm(vectors[i],2)
end
finish=os.clock()
print('take',finish-begin,'seconds')
begin=os.clock()
for i=1,vectors:size(1) do
	expected=torch.dot(vectors[i],vectors[i])
	expected=math.sqrt(expected)
	if math.abs(expected-results[i])>1e-5 then
		print(vectors[i])
		print('expected',expected)
		print('result',results[i])
		print('wrong answer at',i)
		os.exit()
	end
end
finish=os.clock()
print('take',finish-begin,'seconds')
