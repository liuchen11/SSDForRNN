require 'util.randSVD'
require 'math'

matrixNorm={}

function matrixNorm:norm(input,n)
	--calculate the Schatten-n norm of a given matrix
	--the dimension threshold to use randomized SVD is set 100

	local u,s,v
	if input:size(1)<100 or input:size(2)<100 then
		u,s,v=torch.svd(input)
	else
		u,s,v=randSVD:svd(input,100)
	end

	if n==1/0 then 	--infinity norm
		return s[1]
	else
		local sum=0.0
		for i=1,s:size(1) do
			sum=sum+math.pow(s[i],n)
		end
		sum=math.pow(sum,1/n)
		return sum
	end
end