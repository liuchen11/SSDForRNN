import numpy as np

'''
>>> The sparsity of a matrix or vector
>>> The infinity-norm over the 2-norm
'''
def sparsity(array):
	return np.max(array)/np.linalg.norm(array)
