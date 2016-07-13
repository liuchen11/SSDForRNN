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

bsgdt = []
bssdt = []
for i in xrange(len(acc)):
        if mode[i]=='blue':
                bsgdt.append([(val,i) for val in acc[i]])
        else:
                bssdt.append([(val,i) for val in acc[i]])

bsgdr = map(list, zip(*bsgdt))
bssdr = map(list, zip(*bssdt))

bsgds = []
bssds = []

for i in xrange(len(bsgdr)):
        bsgds.append(sorted(bsgdr[i]))

for i in xrange(len(bssdr)):
        bssds.append(sorted(bssdr[i]))

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

order = 2

if order==1:
        bsgdm = []
        bssdm = []

        if len(bsgds)>0:
                for j in xrange(len(bsgds[0])):
                        for i in reversed(xrange(len(bsgds))):
                                bsgdm.append(((bsgds[i])[j])[1])
        if len(bssds)>0:
                for j in xrange(len(bssds[0])):
                        for i in reversed(xrange(len(bssds))):
                                bssdm.append(((bssds[i])[j])[1])

        bsgd = f7(bsgdm)
        bssd = f7(bssdm)                        


elif order==2:
        bsgd = []
        bssd = []

        if len(bsgds)>0:
                i = len(bsgds)-1
                for j in xrange(len(bsgds[0])):
                        bsgd.append(((bsgds[i])[j])[1])
        if len(bssds)>0:
                i = len(bssds)-1
                for j in xrange(len(bssds[0])):
                        bssd.append(((bssds[i])[j])[1])


                        
        
if len(sys.argv)>2:
        nbr = int(sys.argv[2])
else:
        nbr = 5
        
print "\n\n\n\n", bsgd[0:nbr]  
print bssd[0:nbr]
print "\n\n\n"

for i in xrange(min(nbr,len(bsgd))):
	plt.plot(range(1,len(acc[bsgd[i]])+1),acc[bsgd[i]],mode[bsgd[i]])
        print i+1, " (", bsgd[i], ")  : ", ([round(10**s,5) for s in lr_sets[bsgd[i]]]), "\n", acc[bsgd[i]], "\n\n\n"

print "\n"
for i in xrange(min(nbr,len(bssd))):
	plt.plot(range(1,len(acc[bssd[i]])+1),acc[bssd[i]],mode[bssd[i]])
        print i+1, " (", bssd[i], ")  : ", ([round(10**s,5) for s in lr_sets[bssd[i]]]), "\n", acc[bssd[i]], "\n\n\n"

plt.show()
