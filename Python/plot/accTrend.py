import math
import sys

import matplotlib.pyplot as plt

lr_sets=[]
mode=[]
acc=[]
epoch=0

for i in xrange(len(sys.argv)-1):
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
			elif line_mode=='ssd_const_lr':
				mode.append('red')
                        else:
                        	mode.append('black')

                                
if True:
        for i in xrange(len(acc)):
	        plt.plot(range(1,len(acc[i])+1),acc[i],mode[i])
elif False:
        for i in xrange(len(acc)):
	        plt.plot(range(1,len(acc[i])+1),acc[i], label='i = %f'%i)
elif False:
        for j in xrange(len(acc)/7):
                for i in [x+7*j for x in [0,1,2,3,4,5,6]]:
	                plt.plot(range(1,len(acc[i])+1),acc[i], label='i = %f'%i)
                plt.axis([0, 300, 1000, 4000])
                plt.legend(loc=1)
                plt.show()
        for i in [len(acc)+x for x in [-7,-6,-5,-4,-3,-2,-1]]:
	        plt.plot(range(1,len(acc[i])+1),acc[i], label='i = %f'%i)
        plt.axis([0, 300, 1000, 4000])
        plt.legend(loc=1)
        plt.show()
        
elif True:
        #17 29 48 51 52
        #plt.plot(range(1,301),acc[1][0:300], 'blue', linewidth=3, label='SGD, constant learning rate')
        #plt.plot(range(1,251),acc[0][0:250], 'red', linewidth=5, label='SSD, constant learning rate')
        #plt.plot(range(1,151),acc[3][0:150], 'red', linewidth=5, label='SSD, constant learning rate')
        #plt.plot(range(1,301),acc[2][0:300], 'blue', linewidth=3, linestyle='--', label='SGD, decreasing learning rate')
        #plt.subplot(223)

        
        #plt.plot(range(1,251),acc[25][0:250], 'red', linewidth=5, label='SSD, constant learning rate')
        #plt.plot(range(1,len(acc[29])+1),acc[29], mode[29], linewidth=3)
        j = -5
        for i in [x+j for x in [17,29,51]]:
                plt.plot(range(1,len(acc[i])+1),acc[i], mode[i], linewidth=3)
        
elif True:
        figure = plt.figure()
        figure.subplots_adjust(left = 0.12, bottom = 0.1, right = 0.95, top = 0.95, wspace = 0.35, hspace = 0)
        plt.subplot(121)
        plt.plot(range(1,301),acc[16][0:300], 'blue', linewidth=3, label='SGD, constant LR')
        plt.plot(range(1,151),acc[29][0:150], 'red', linewidth=5, label='SSD, constant LR')
        plt.plot(range(1,301),acc[2][0:300], 'blue', linewidth=3, linestyle='--', label='SGD, decreasing LR')
        plt.xlabel('Epoch', fontsize=17)
        plt.axis([0, 300, 1000, 4000])
        plt.ylabel('Training Error', fontsize=17)

        plt.subplot(122)
        plt.plot(range(1,301),acc[16+30][0:300], 'blue', linewidth=3, label='SGD, constant LR')
        plt.plot(range(1,151),acc[29+30][0:150], 'red', linewidth=5, label='SSD, constant LR')
        plt.plot(range(1,301),acc[2+30][0:300], 'blue', linewidth=3, linestyle='--', label='SGD, decreasing LR')
        plt.xlabel('Epoch', fontsize=17)
        plt.axis([0, 300, 1000, 4000])
        plt.ylabel('Testing Error', fontsize=17)
        plt.legend(loc=1)

        
else:
        #plt.subplot(223)
        
        plt.plot(range(1,501),acc[14][0:500], 'blue', linewidth=3, label='SGD, constant learning rate')
        plt.plot(range(1,251),acc[29][0:250], 'red', linewidth=5, label='SSD, constant learning rate')
        plt.plot(range(1,501),acc[34][0:500], 'blue', linewidth=3, linestyle='--', label='SGD, decreasing learning rate')

#plt.title('TEST', fontsize=23)

plt.show()

