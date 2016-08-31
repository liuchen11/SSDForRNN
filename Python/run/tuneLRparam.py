import sys
import os
import random
import math

import numpy as np

'''
>>> Try a specific learning rate setting for serveral times
>>> mode: can be 'all' or a specified optimzer like 'sgd_const_lr' etc, or the following groups of optimizers : 'ssd', sgd', 'const_lr', 'rms', 'adagrad'
>>> U_lr, W_lr, V_lr, s_lr: float. Specified learning rate for different matrices
>>> points: integer. Specify how many times we try for this learning rate setting
>>> outfile: string, optional. This parameter set the output file and writing mode(append or rewrite). The default is "a" and "log"
'''

if len(sys.argv)<7:
    print 'Usage: python tuneLRparam.py <mode> <U_lr> <W_lr> <V_lr> <s_lr> <points> (<a/w> <outfile>)'
    exit(0)

mode=sys.argv[1]
p1=float(sys.argv[2])
p2=float(sys.argv[3])
p3=float(sys.argv[4])
p4=float(sys.argv[5])
points=int(sys.argv[6])

outfile='log'
if len(sys.argv)>=9:
    if sys.argv[8][-1]=='/':
        print 'Invalid output dictionary: %s'%sys.argv[8]
        exit(0)
    outfile=sys.argv[8]
    while outfile.count('/')>0:
        outfile=outfile[outfile.index('/')+1:]

print 'Output File Path=results/%s'%outfile
file_mode=sys.argv[7] if len(sys.argv)>=8 else 'a'

if os.path.exists('../results')==False:
    os.system('mkdir ../results')
outfile='../results/'+outfile
lr=[p1,p2,p3,p4]

if mode=='all' or mode=='sgd_const_lr' or mode=='sgd' or mode=='const_lr':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('sgd_const_lr',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='ssd_const_lr' or mode=='ssd' or mode=='const_lr':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('ssd_const_lr',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='sgd_rms' or mode=='sgd' or mode=='rms':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('sgd_rms',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='ssd_rms' or mode=='ssd' or mode=='rms':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('ssd_rms',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='sgd_adagrad' or mode=='sgd' or mode=='adagrad':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('sgd_adagrad',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))

if mode=='all' or mode=='ssd_adagrad' or mode=='ssd' or mode=='adagrad':
    for i in xrange(points):
        os.system('python ../tasks/atisLabel.py %s %f %f %f %f %s %s'
                  %('ssd_adagrad',lr[0],lr[1],lr[2],lr[3],file_mode,outfile))
