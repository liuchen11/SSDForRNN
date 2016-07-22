import math
import sys

import matplotlib.pyplot as plt
from collections import OrderedDict

lr_sets=[]
mode=[]
acc=[]
epoch=0

begin=1
max_epoch=-1	#max_epoch control the maximum length of plotted epoches, negative value means no limit
if len(sys.argv)>3 and sys.argv[1]=='-n':
	max_epoch=int(sys.argv[2])
	begin=3


for i in xrange(len(sys.argv)-begin):
	with open(sys.argv[i+begin],'r') as fopen:
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
			line_acc=[]
			for i in xrange(len(results_part)-1):
				line_acc.append(float(results_part[i+1]))
			lr_sets.append([line_U_lr,line_W_lr,line_V_lr,line_s_lr])
			acc.append(line_acc)
			if line_mode=='sgd_const_lr':
				mode.append('blue')
			elif line_mode=='ssd_const_lr':
				mode.append('red')
			else
				mode.append('black')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
color2label={'blue':'SGD','red':'SSD','black':'RMS'}
for i in xrange(len(acc)):
	epoch=len(acc[i]) if max_epoch<0 else min(max_epoch,len(acc[i]))
	ax.plot(range(1,epoch+1),acc[i][:epoch],mode[i],label=color2label[mode[i]])

handles,labels=ax.get_legend_handles_labels()
by_label=OrderedDict(zip(labels,handles))
ax.legend(by_label.values(),by_label.keys())
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()