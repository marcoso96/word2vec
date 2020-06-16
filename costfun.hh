#include "matrix.hh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

class W2VCost {
private:

    void softLoss(Matrix &outsideVecs, Matrix &centerVec, Matrix& Y_est, float* loss, int outsideVecIdx);
    void gradCenter(Matrix &outsideVecs, Matrix &Y_est, Matrix &gradCenter);
    void gradOutside(Matrix &centerVec, Matrix &Y_est, Matrix &gradOutside);

public: 

    void lossAndGrad(Matrix &centerVec, Matrix &outsideVecs, int outsideVecIdx, float *loss, Matrix &gradCenter, Matrix &gradOutside);
};