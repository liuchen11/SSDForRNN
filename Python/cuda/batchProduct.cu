// function: oneXn
// m1 of shape H*L, m2 of shape N*L*M, m3 of shape N*H*M
// N blocks, 2 dimensional block size
__global__ void oneXn(float* m1, float* m2, float* m3,
    int size_n, int size_l, int size_h, int size_m){
    int n=size_n,l=size_l, h=size_h, m=size_m;

    for(int h_index=threadIdx.x;h_index<h;h_index+=blockDim.x){
        for(int m_index=threadIdx.y;m_index<m;m_index+=blockDim.y){
            float value=0.0;
            for(int l_index=0;l_index<l;l_index++){
                float x=m1[h_index*l+l_index];
                float y=m2[blockIdx.x*l*m+l_index*m+m_index];
                value+=x*y;
            }
            m3[blockIdx.x*h*m+h_index*m+m_index]=value;
        }
    }
}

// function: nXone
// m1 of shape N*H*L, m2 of shape L*M, m3 of shape N*H*M
// N blocks, 2 dimensional block size
__global__ void nXone(float* m1, float* m2, float* m3,
    int size_n, int size_l, int size_h, int size_m){
    int n=size_n, l=size_l, h=size_h, m=size_m;

    for(int h_index=threadIdx.x;h_index<h;h_index+=blockDim.x){
        for(int m_index=threadIdx.y;m_index<m;m_index+=blockDim.y){
            float value=0.0;
            for(int l_index=0;l_index<l;l_index++){
                float x=m1[blockIdx.x*h*l+h_index*l+l_index];
                float y=m2[l_index*m+m_index];
                value+=x*y;
            }
            m3[blockIdx.x*h*m+h_index*m+m_index]=value;
        }
    }
}