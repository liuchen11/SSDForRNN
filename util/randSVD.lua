randSVD={}

function randSVD:svd(input,d)
	--randomized singular value decomposition
	--input: input matrix
	--d: the num of estimated singular values

	local trans=false
	local X=input
	if input:size(2)>input:size(1) then
		trans=true
		X=X:t()
	end

	local omega=torch.randn(X:size(2),d)
	local Q,R=torch.qr(X*omega)
	local B=Q:t()*X
	local u,s,v=torch.svd(B)
	u=Q*u
	--s is a vector and u,v are matrices
	--elements in s are sortes from largest to smallest

	if trans then
		return v,s,u
	else
		return u,s,v
	end
end