#include <math.h>

void sigmoid_unit(float v){
    return 1.0/(1.0+exp(-v));
}

void dsigmoid_unit(float v){
    return sigmoid_unit(v)*(1.0-sigmoid_unit(v));
}

// function sigmoid
// 1 block, 1 dimensional block size
__global__ void sigmoid(float* m1, float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=sigmoid_unit(m1[index]);
}

// function sigmoid
// 1 block, 1 dimensional block size
__global__ void dsigmoid(float* m1,float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=dsigmoid_unit(m1[index]);
}