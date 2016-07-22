import sys
import os
import random
import math

import numpy as np

'''
>>> Try different lr randomly to tune lr
>>> mode: can be 'all' or a specified optimizer like 'sgd_const_lr' etc
>>> start, end: the lower and upper bound of lr, the logarithm of lr for each parameter set is sampled uniformly i.i.d
>>> points: integer. Specify how many sets of lr we try in total
>>> outfile: string, optional. This parameter set the output file and writing mode(append or rewrite), the default value is "a" and "log"
'''

if len(sys.argv)<5:
	print 'Usage: python tuneLR.py <mode> <start> <end> <points> (<a/w> <outfile>)'
	exit(0)

mode=sys.argv[1]
p1=float(sys.argv[2])
p2=float(sys.argv[3])
points=int(sys.argv[4])

outfile='log'
if len(sys.argv)>=7:
	if sys.argv[6][-1]=='/':
		print 'Invalid output dictionary: %s'%sys.argv[6]
		exit(0)
	outfile=sys.argv[6]
	while outfile.count('/')>0:
		outfile=outfile[outfile.index('/')+1:]

print 'Output File Path=results/%s'%outfile
file_mode=sys.argv[5] if len(sys.argv)>=6 else 'a'

if os.path.exists('../results')==False:
	os.system('mkdir ../results')
outfile='../results/'+outfile

if mode=='all' or mode=='sgd_const_lr':
	rate=p2/p1
	for i in xrange(points):
		lr=np.zeros(4)
		for j in xrange(4):
			lr[j]=p1*rate**random.random()
		os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'%(
			'sgd_const_lr',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='ssd_const_lr':
	rate=p2/p1
	for i in xrange(points):
		lr=np.zeros(4)
		for j in xrange(4):
			lr[j]=p1*rate**random.random()
		os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'%(
			'ssd_const_lr',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='ssd_rms':
	rate=p2/p1
	for i in xrange(points):
		lr=np.zeros(4)
		for j in xrange(4):
			lr[j]=p1*rate**random.random()
		os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'%(
			'ssd_rms',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))
