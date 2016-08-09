from multiprocessing import Pool
import os
import sys

'''
>>> Launch several trainings with different parameters in parallel
>>> 1st argument : number of processus to start
>>> 2nd, 3rd, 4th, ... argument = integer that defines a set of parameters to use
'''




'''
>>> Start a training with random parameters
'''
def tunem():
	os.system('python tuneLR.py all 0.005 0.01 1')


'''
>>> Start a training with chosen parameters
>>> n: integer which defines one set of parameters
'''
def tunec(n):
	param = getparam(n)
	os.system('python tuneLRparam.py %s %f %f %f %f 1'%(param['mode'], param['U'], param['W'], param['V'], param['s']))





'''
>>> Return a set of chosen parameters
>>> n: integer which defines the set of parameters
'''
def getparam(n):
	allmode = []
	for i in xrange(10):
		allmode = allmode + ['sgd_const_lr']
	for i in xrange(10):
		allmode = allmode + ['ssd_const_lr']
	allparam = []
	allparam = allparam + [[0.0061, 0.0053, 0.009, 0.0054]]
	allparam = allparam + [[0.0054, 0.005, 0.0092, 0.0081]]
	allparam = allparam + [[0.0074, 0.0055, 0.0098, 0.0094]]
	allparam = allparam + [[0.0053, 0.0058, 0.0084, 0.007]]
	allparam = allparam + [[0.0065, 0.0085, 0.0086, 0.0051]]
	allparam = allparam + [[0.0063, 0.0052, 0.0076, 0.0065]]
	allparam = allparam + [[0.006, 0.007, 0.0091, 0.0054]]
	allparam = allparam + [[0.0067, 0.0081, 0.0092, 0.0091]]
	allparam = allparam + [[0.0055, 0.0057, 0.0079, 0.0091]]
	allparam = allparam + [[0.0052, 0.0077, 0.0094, 0.0077]]
	allparam = allparam + [[0.0066, 0.0054, 0.0093, 0.0083]]
	allparam = allparam + [[0.0095, 0.0063, 0.0098, 0.0058]]
	allparam = allparam + [[0.0054, 0.0066, 0.0092, 0.007]]
	allparam = allparam + [[0.0092, 0.0063, 0.0099, 0.006]]
	allparam = allparam + [[0.0095, 0.0061, 0.0088, 0.006]]
	allparam = allparam + [[0.0055, 0.0061, 0.0066, 0.0072]]
	allparam = allparam + [[0.007, 0.0086, 0.0085, 0.0074]]
	allparam = allparam + [[0.0058, 0.0055, 0.0093, 0.009]]
	allparam = allparam + [[0.0091, 0.0084, 0.0091, 0.0057]]
	allparam = allparam + [[0.009, 0.0097, 0.0091, 0.0053]]
	param = {}
	param['mode'] = allmode[n]
	param['U'] = allparam[n][0]
	param['W'] = allparam[n][1]
	param['V'] = allparam[n][2]
	param['s'] = allparam[n][3]
	return param




if len(sys.argv)<2:
	Nprocess = 1
else:
	Nprocess = int(sys.argv[1])
if Nprocess < 1:
	Nprocess = 1

whichparams = []
if Nprocess < len(sys.argv)-1:
	for i in sys.argv[2:Nprocess+1]:
		whichparams.append(int(i))



pool = Pool()

if Nprocess==1:
	if len(whichparams)>=1:
		tunec(whichparams[0])
	else:
		tunem()
else:
	result = []
	answer = []
	for i in xrange(Nprocess):
		result.append(0);
		answer.append(0);
	if len(whichparams)>=1:
		for i in xrange(Nprocess):
			result[i] = pool.apply_async(tunec, args=(witchparams[i],))
	else:
		for i in xrange(Nprocess):
			result[i] = pool.apply_async(tunem)

	for i in xrange(Nprocess):
		answer[i] = result[i].get(timeout=1000000)
