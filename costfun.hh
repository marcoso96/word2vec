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

class W2VCost {
private:

    float *Y_est;
    float *grad_center;
    float *grad_outside;

    float *loss;

    int context;
    int embed_size;
    int batch_size;

    void softLoss(float *outsideVecs, float *centerVec, int outIdx);
    void gradCenter(float *outsideVecs);
    void gradOutside(float *centerVec);

public: 

    W2VCost(int embed_size, int batch_size);
    ~W2VCost();
    
    void lossAndGrad(float *centerVec, float *outsideVecs, int batch_size);
};