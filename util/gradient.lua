gradient={}

function gradient:dsigmoid(value)
	local sig=nn.sigmoid()(value)
	return torch.cmul(sig,torch.ones(sig:size())-sig)
end

function gradient:drelu(value)
	
end