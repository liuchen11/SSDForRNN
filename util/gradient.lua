require 'math'
--max dsigmoid=0.25 sigmoid=0.5
--max drelu=1
--max ddsigmoid=\sqrt{3}/18 sigmoid=\frac{3-\sqrt{3}}{6}
gradient={max_dsigmoid=0.25,max_drelu=1,max_ddsigmoid=math.sqrt(3)/18}

function sigmoidElement(value)
	--sigmoid function for a given scalar
	return 1.0/(1+math.exp(-value))
end

function dsigmoidElement(value)
	--gradient of sigmoid function for a given scalar
	local sig=sigmoidElement(value)
	return sig*(1-sig)
end

function gradient:dsigmoid(value)
	--gradient of sigmoid function
	--value is a matrix
	local gvalue=torch.Tensor(value:size()):copy(value)	--deep copy
	return gvalue:apply(dsigmoidElement)
end

function relu(value)
	--relu function for a given scalar
	if value>0 then
		return value
	else
		return 0
	end
end

function grelu(value)
	--gradient of relu when value is a scalar
	if value>0 then
		return 1
	else
		return 0
	end
end

function gradient:drelu(value)
	--gradient of relu function
	--value is a matrix
	gvalue=torch.Tensor(value:size()):copy(value)	--deep copy, we shouldn't change the value
	return gvalue:apply(grelu)
end