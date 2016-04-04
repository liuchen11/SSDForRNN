require 'math'

vectorNorm={}

function vectorNorm:norm(input,n)
	--calculate the n-norm of a given vector
	--if the input is a matrix, we will treat it like a vector

	local s=input:storage()
	local num=input:elementSize()
	if n==1/0 then
		local max=0
		for i=1,num do
			if max<math.abs(s[i]) then
				max=math.abs(s[i])
			end
		end
		return max
	else
		local sum=0
		for i=1,num do
			sum=sum+math.abs(math.pow(s[i],n))
		end
		sum=math.pow(sum,1/n)
		return sum
	end
end