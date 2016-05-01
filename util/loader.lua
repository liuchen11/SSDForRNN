loader={}

function loader:loadData(fileName)
	--load trainset/testset data/label
	--output a 2d array
	local ret={}
	for line in io.lines(fileName) do
		local sentence=line:split(',')
		table.insert(ret,sentence)
	end
	return ret
end

function loader:loadDict(fileName)
	--load a dictionary from file
	local ret={}
	for line in io.lines(fileName) do
		local parts=line:split(':')
		local index=parts[1]
		local vector=torch.Tensor(parts[2]:split(','))
		ret[index]=vector
	end
	return ret
end