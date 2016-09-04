#include <math.h>

// function add
// m1, m2 are vectors, matrices or tensors of the same size
// 1 block, 1 dimensional block size
__global__ void add(float* m1, float* m2, float* m3, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m3[index]=m1[index]+m2[index];
}

// function minus
// m1, m2 are vectors, matrices or tensors of the same size
// 1 block, 1 dimensional block size
__global__ void minus(float* m1, float* m2, float* m3, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m3[index]=m1[index]-m2[index];
}

// function mul
// 1 block, 1 dimensional block size
__global__ void mul(float* m1, float factor, float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=m1[index]*factor;
}

// function div
// 1 block, 1 dimensional block size
__global__ void div(float* m1, float divsor, float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=m1[index]/divsor;
}

// function pow
// 1 block, 1 dimensional block size
__global__ void pow(float* m1, float power, float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=pow(m1[index],power);
}

// function sqrt
// 1 block, 1 dimensional block size
__global__ void sqrt(float* m1, float* m2, int n){
    for(int index=threadIdx.x;index<n;index+=blockDim.x)
        m2[index]=sqrt(m1[index]);
}

// function dot
// m1 of shape H*L, m2 of shape L*M and m3 of shape H*M
// 1 block, 2 dimensional block size
__global__ void dot(float* m1, float* m2, float* m3,
    int size_h, int size_l, int size_m){
    int h=size_h, l=size_l, m=size_m;

    for(int h_index=threadIdx.x;h_index<h;h_index+=blockDim.x){
        for(int m_index=threadIdx.y;m_index<m;m_index+=blockDim.y){
            float value=0.0;
            for(int l_index=0;l_index<l;l_index++){
                float x=m1[h_index*l+l_index];
                float y=m2[l_index*m+m_index];
                value+=x*y;
            }
            m3[h_index*m+m_index]=value;
        }
    }
}

