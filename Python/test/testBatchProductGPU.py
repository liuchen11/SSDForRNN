import sys
sys.path.insert(0,'./util')

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import time
import batchProduct
import loader

x1=np.random.randn(100,200,100)
x2=np.random.randn(100,200)

begin=time.time()
result_cpu=batchProduct.nXone(x1,x2)
end=time.time()
print 'time on CPU:',end-begin

source=loader.loadCudaFunc('./cuda/batchProduct.cu',['nXone'])%{
    'MATRIX_SIZE_N':100,'MATRIX_SIZE_L':100,'MATRIX_SIZE_H':200,'MATRIX_SIZE_M':200
}
mod=SourceModule(source)
function_gpu=mod.get_function('nXone')

x1=x1.astype(np.float32)
x2=x2.astype(np.float32)
result=np.zeros([100,200,200],dtype=np.float32)
begin=time.time()
x1_gpu=cuda.mem_alloc(x1.nbytes)
x2_gpu=cuda.mem_alloc(x2.nbytes)
result_gpu=cuda.mem_alloc(result.nbytes)
cuda.memcpy_htod(x1_gpu,x1)
cuda.memcpy_htod(x2_gpu,x2)

function_gpu(x1_gpu,x2_gpu,result_gpu,block=(10,10,1),grid=(100,1))
cuda.memcpy_dtoh(result,result_gpu)
end=time.time()
print 'time on GPU:',end-begin

print np.sum(np.abs(result_cpu-result))