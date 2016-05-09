require 'math'

vectorNorm={}

function vectorNorm:norm(input,n)
	--calculate the n-norm of a given vector
	--the input must be a 1D vector

	local num=input:size(1)
	if n==1/0 then
		local max=0
		for i=1,num do
			if max<math.abs(input[i]) then
				max=math.abs(input[i])
			end
		end
		return max
	end
	if n==2 then
		return math.sqrt(torch.dot(input,input))		--much faster
	else
		local sum=0
		for i=1,num do
			sum=sum+math.abs(math.pow(input[i],n))
		end
		sum=math.pow(sum,1/n)
		return sum
	end
end