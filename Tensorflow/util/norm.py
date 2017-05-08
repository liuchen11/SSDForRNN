import tensorflow as tf

'''
>>> p-norm
>>> order: int or list of int
>>> return: tf.tensor or list of tf.tensor
'''
def p_norm(tensor,order):
    if type(order) in [int,float]:
        return tf.norm(tensor,ord=order)
    elif type(order) in [list,tuple]:
        return [tf.norm(tensor,ord=order_item) for order_item in order]
    else:
        raise ValueError('Unrecognized order of p_norm: %s'%str(order))


'''
>>> shatten norm of matrix
>>> order: int of list of int
>>> return: tf.tensor, tf.tensor or list of tf.tensor
'''
def s_norm(tensor,order):
    s,U,V=tf.svd(tensor,full_matrices=False)
    result=None
    if type(order) in [int,float]:
        result=tf.norm(s,ord=order)
    elif type(order) in [list,tuple]:
        result=[tf.norm(s,ord=order_item) for order_item in order]
    else:
        raise ValueError('Unrecognized order of s_norm: %s'%str(order))
    return s,result
