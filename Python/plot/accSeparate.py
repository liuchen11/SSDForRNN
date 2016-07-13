import math
import sys

import matplotlib.pyplot as plt

lr_sets=[]
mode=[]
acc=[]
epoch=0

i = 0
with open(sys.argv[i+1],'r') as fopen:
	content=fopen.read()
	lines=content.split('\n')
	for j in xrange(len(lines)/3):
		params=lines[3*j]
		results=lines[3*j+1]
		params_part=params.split(',')
		results_part=results.split('|')
		epoch=len(results_part)-1
		line_mode=params_part[0].split('=')[1]
		line_U_lr=math.log10(float(params_part[1].split('=')[1]))
		line_W_lr=math.log10(float(params_part[2].split('=')[1]))
		line_V_lr=math.log10(float(params_part[3].split('=')[1]))
		line_s_lr=math.log10(float(params_part[4].split('=')[1]))
		line_acc=[]
		for i in xrange(len(results_part)-1):
			line_acc.append(float(results_part[i+1]))
		if line_acc[0]<line_acc[-1]:
			continue
                lr_sets.append([line_U_lr,line_W_lr,line_V_lr,line_s_lr])
		acc.append(line_acc)
		if line_mode=='sgd_const_lr':
			mode.append('blue')
		else:
			mode.append('red')

if len(sys.argv)>2:
        deb = 0
        nbr = int(sys.argv[2])
        while deb*nbr<len(acc):
                for j in xrange(nbr):
                        if deb*nbr+j<len(acc):
                            plt.plot(range(1,epoch+1),acc[deb*nbr+j],mode[deb*nbr+j])
                            
                plt.show()
                deb=deb+1
else:
        for i in xrange(len(acc)):
	        plt.plot(range(1,epoch+1),acc[i],mode[i])
        plt.show()
