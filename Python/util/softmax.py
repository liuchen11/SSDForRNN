import numpy as np

def softmax(v):
    sv=np.exp(v)
    sv=sv/np.sum(sv,axis=0)
    return sv