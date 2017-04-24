import tensorflow as tf

'''
>>> sharp operator for schatten-inf norm
>>> input: tf.tensor, input 2d matrix of tensorflow format
>>> output: result of applying sharp operator
'''
def sharp(input):
    s,U,V=tf.svd(input,full_matrices=False)
    return tf.matmul(U,V)*tf.reduce_sum(s)
