import sys
sys.path.insert(0,'util/')
import numpy as np
import matplotlib.pyplot as plt

import gradient

marks=np.arange(-2,2,0.001)
sigmoid_result=gradient.sigmoid(marks)
dsigmoid_result=gradient.dsigmoid(marks)
relu_result=gradient.relu(marks)
drelu_result=gradient.drelu(marks)
tanh_result=gradient.tanh(marks)
dtanh_result=gradient.dtanh(marks)

sigmoid_handle,=plt.plot(marks,sigmoid_result,color='red',label='sigmoid')
dsigmoid_handle,=plt.plot(marks,dsigmoid_result,color='orange',label='dsigmoid')
relu_handle,=plt.plot(marks,relu_result,color='black',label='relu')
drelu_handle,=plt.plot(marks,drelu_result,color='yellow',label='drelu')
tanh_handle,=plt.plot(marks,tanh_result,color='green',label='tanh')
dtanh_handle,=plt.plot(marks,dtanh_result,color='blue',label='dtanh')

plt.legend([sigmoid_handle,dsigmoid_handle,relu_handle,drelu_handle,tanh_handle,dtanh_handle])
plt.show()