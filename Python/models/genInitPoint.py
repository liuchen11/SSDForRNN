import os
import sys
sys.path.insert(0,'util/')

import RNNs
import xmlParser

if len(sys.argv)<2:
    print 'Usage: python genInitPoint.py <xml>'
    exit(0)

config=xmlParser.parse(sys.argv[1],flat=False)
neurons=config['neurons']
nonlinearity=config['nonlinearity']
output_file=config['output_file'] if config.has_key('output_file') else '.'.join(sys.argv[1].split('.')[:-1])+'.pkl'

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

rnn=RNNs.RNNs(neurons,nonlinearity)
rnn.save(output_file,testOnly=True)
print 'model saved in %s'%output_file
