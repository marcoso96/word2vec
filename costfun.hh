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

void print_matrix(double *Mat, int Mat_height, int Mat_width);
class W2VCost {
private:

    double *Y_est;
    double *grad_center;
    double *grad_outside;
    double *loss;

    int embed_size;
    int vocab_size;
    int out_offset;

    int iteration;
    double lr;
    int batch_size = 50;

    void softLoss(double *wordVecs, int centerVecIdx);
    void gradCenter(double *outsideVecs);
    void gradOutside(double *centerVec);

    void updateOutside(double* outsideVecs);
    void updateCenter(double* centerVec);

public: 

    W2VCost(int embed_size, int vocab_size , double lr);
    ~W2VCost();
    
    void lossAndGrad(double* wordVecs, int* outsideIdxs,  int centerIdx, int context_size);
    void updateGradients(double* wordVecs, int centerIdx);
};