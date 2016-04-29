require 'util.gradient'

print(gradient.max_dsigmoid)
print(gradient.max_ddsigmoid)
print(gradient.max_drelu)

matrix=torch.randn(4,4)
print(matrix)
print(gradient:dsigmoid(matrix))
print(gradient:ddsigmoid(matrix))
print(gradient:drelu(matrix))