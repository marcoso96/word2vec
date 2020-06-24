#include "matrix.hh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_matrix(float *Mat, int Mat_height, int Mat_width);
class W2VCost {
private:

    float *Y_est;
    float *grad_center;
    float *grad_outside;
    float *loss;

    int context;
    int embed_size;
    int batch_size;
    int vocab_size;
    int out_offset;

    int iteration;
    int lr;

    void softLoss(float *wordVecs, int centerVecIdx);
    void gradCenter(float *outsideVecs);
    void gradOutside(float *centerVec);

    void updateOutside(float* wordVecs);
    void updateCenter(float* wordVecs);

public: 

    W2VCost(int embed_size, int batch_size, int vocab_size, , int lr);
    ~W2VCost();
    
    void lossAndGrad(float* wordVecs, int* outsideIdxs,  int centerIdx, int batch_size);
    void updateGradients(float* wordVecs, int centerIdx);
};