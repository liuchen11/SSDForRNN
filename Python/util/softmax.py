import numpy as np

def softmax(v):
    mv=v-np.max(v)  # to avoid numeral overflow
    sv=np.exp(mv)
    sv=sv/np.sum(sv,axis=0)
    return sv