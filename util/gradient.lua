gradient={max_dsigmoid=0.25,max_drelu=1}

function gradient:dsigmoid(value)
	--gradient of sigmoid function
	--value is a matrix
	local sig=nn.sigmoid()(value)
	return torch.cmul(sig,torch.ones(sig:size())-sig)
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