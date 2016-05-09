require 'util.randSVD'
require 'math'

sharp={}

function sharp:sharp(input)
	--apply shatten-infinity norm-based #-operator to an input matrix
	--s#=argmax {<s,x>-1/2|x|^2}
	--the dimension threshold to use the randomized SVD is set to be 100

	local u,s,v
	if input:size(1)<100 or input:size(2)<100 then
		u,s,v=torch.svd(input)
	else
		u,s,v=randSVD:svd(input,100)
	end

	--x=usv^T => x#=|s|_{s^1}uv^T
	local r=torch.mm(u,v[{{},{1,u:size(2)}}]:t()):mul(s:sum())
	return r
end