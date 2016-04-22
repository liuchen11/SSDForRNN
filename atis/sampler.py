import numpy as np
import random
import sys

if __name__=='__main__':
	if len(sys.argv)!=6:
		print 'Usage: python sampler.py <input1> <input2> <percentage> <output1> <output2>'
		exit(0)

	percentage=True
	radio=float(sys.argv[3])
	number=0
	if radio>1:
		percentage=False
		number=int(sys.argv[3])
	elif number<=0:
		print 'invalid percentage!'
		exit(0)

	input1=sys.argv[1]
	input2=sys.argv[2]
	output1=sys.argv[4]
	output2=sys.argv[5]

	trainX=[]
	trainY=[]
	with open(input1,'r') as fopen:
		for line in fopen:
			trainX.append(line)

	with open(input2,'r') as fopen:
		for line in fopen:
			trainY.append(line)

	instances=len(trainX)
	orders=range(instances)
	random.shuffle(orders)

	if percentage==True:
		number=int(instances*radio)

	with open(output1,'w') as fopen1:
		with open(output2,'w') as fopen2:
			for i in xrange(number):
				fopen1.write(trainX[orders[i]])
				fopen2.write(trainY[orders[i]])

