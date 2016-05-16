import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

lr_sets=[]
mode=[]
acc=[]

for i in xrange(len(sys.argv)-1):
	with open(sys.argv[i+1],'r') as fopen:
		content=fopen.read()
		lines=content.split('\n')
		for j in xrange(len(lines)/3):
			params=lines[3*j]
			results=lines[3*j+1]
			params_part=params.split(',')
			results_part=results.split('|')
			line_mode=params_part[0].split('=')[1]
			line_U_lr=math.log10(float(params_part[1].split('=')[1]))
			line_W_lr=math.log10(float(params_part[2].split('=')[1]))
			line_V_lr=math.log10(float(params_part[3].split('=')[1]))
			line_s_lr=math.log10(float(params_part[4].split('=')[1]))
			line_result='%.2f'%(float(results_part[-1]))
			lr_sets.append([line_U_lr,line_W_lr,line_V_lr,line_s_lr])
			acc.append(line_result)
			if line_mode=='sgd_const_lr':
				mode.append('blue')
			else:
				mode.append('red')

lr_sets=np.asarray(lr_sets)

# fig=plt.figure()
# ax=fig.gca(projection='3d')
# ax.scatter(lr_sets[:,0],lr_sets[:,1],lr_sets[:,2],mode)
# for i,txt in enumerate(acc):
# 	ax.text(lr_sets[i,0],lr_sets[i,1],lr_sets[i,2],txt,color=mode[i])
# plt.show()

pca=decomposition.PCA()
pca.n_components=3
lr_sets=pca.fit_transform(lr_sets)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.scatter(lr_sets[:,0],lr_sets[:,1],lr_sets[:,2],mode)
for i,txt in enumerate(acc):
	ax.text(lr_sets[i,0],lr_sets[i,1],lr_sets[i,2],txt,color=mode[i])
plt.show()

# pca=decomposition.PCA()
# pca.n_components=2
# lr_sets=pca.fit_transform(lr_sets)
# fig,ax=plt.subplots()
# ax.scatter(lr_sets[:,0],lr_sets[:,1],color=mode)
# for i,txt in enumerate(acc):
# 	ax.text(lr_sets[i,0],lr_sets[i,1],txt,color=mode[i])
# plt.show()

