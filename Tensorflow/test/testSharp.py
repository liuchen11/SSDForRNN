import sys
sys.path.insert(0,'./util')
import numpy as np
import tensorflow as tf

import sharp

def np_sharp(input):
    U,s,V=np.linalg.svd(input,full_matrices=False)
    return np.dot(U,V)*np.sum(s)

def vec_norm(input,n):
    # vectorize
    inp=np.array(input).reshape(-1)

    if n==np.inf:
        return np.max(inp)
    elif n==2:
        p=np.dot(inp,inp)
        return math.sqrt(p)
    return np.linalg.norm(inp,n)

def norm(input,n):
    U,s,V=np.linalg.svd(input)
    return vec_norm(s,n)

def func(M1,M2):
    dot_product=np.dot(M1.reshape(-1),M2.reshape(-1))
    normalize=norm(M2,np.inf)**2/2
    return dot_product-normalize

M_value=np.random.randn(200,100)
M=tf.Variable(M_value,dtype=tf.float32)
M_sharp=sharp.sharp(M)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    M_sharp_value,=sess.run([M_sharp])
    # M_sharp_value=np_sharp(M_value)
    max_V=func(M_value,M_sharp_value)
    for i in xrange(100):
        rand=np.random.randn(200,100)
        V=func(M_value,rand)
        try:
            sys.stdout.write('Pass %d/%d\r'%(i+1,100))
            assert(V<=max_V)
            sys.stdout.flush()
        except:
            print('V=%f'%V)
            print('max_V=%f'%max_V)
            exit(0)

