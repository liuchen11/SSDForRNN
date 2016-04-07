require 'util.randSVD'
require 'math'

matrixNorm={}

function matrixNorm:norm(input,n,d)
	--calculate the Schatten-n norm of a given matrix
	--input:matrix
	--n: n-norm
	--d: approximate dimension if d is non-positive, use precise SVD

	local u,s,v
	if d>0 and d<input:size(1) and d<input:size(2) then
		u,s,v=randSVD:svd(input,d)
	else
		u,s,v=torch.svd(input)
	end

	if n==1/0 then 	--infinity norm
		return s[1]
	end
	if n==2 then
		return math.sqrt(torch.dot(s,s))	--dot product is much faster
	else
		local sum=0.0
		for i=1,s:size(1) do
			sum=sum+math.pow(s[i],n)
		end
		sum=math.pow(sum,1/n)
		return sum
	end
end