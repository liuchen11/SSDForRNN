import numpy as np
import sys

from sklearn import decomposition

if __name__=='__main__':
	if len(sys.argv)!=4:
		print('Usage: python PCA.py <input> <target_dim> <output>')
		exit(0)

	target_dim=int(sys.argv[2])
	inFile=sys.argv[1]
	outFile=sys.argv[3]

	vectors=[]
	index=[]
	src_dim=-1
	with open(inFile,'r') as fopen:
		num=0
		for line in fopen:
			num+=1
			parts=line.split(':')
			index.append(parts[0])
			elements=parts[1].split(',')
			if src_dim>0 and len(elements)!=src_dim:
				print 'dimension don\'t agree between line %d and previuous lines'%num
				print '%d expected but %d detected'%(src_dim,len(elements))
				exit(0)
			else:
				src_dim=len(elements)
			add=[]
			for entry in elements:
				add.append(float(entry))
			vectors.append(add)
	print 'raw vectors loaded!'

	matrix=np.array(vectors)
	pca=decomposition.PCA()
	pca.n_components=target_dim
	matrix=pca.fit_transform(matrix)
	print 'pca completed!'

	with open(outFile,'w') as fopen:
		num=0
		for line in matrix:
			fopen.write(str(index[num])+':'+str(line[0]))
			for i in xrange(len(line)-1):
				fopen.write(','+str(line[i+1]))
			fopen.write('\n')
			num+=1
	print 'new vectors created!'