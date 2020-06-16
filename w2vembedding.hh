#include "matrix.hh"
#include <iostream>
#include <curand.h>
#include <cuda.h>

class W2VEmbedding{

private :

    void initW2V();
    int context;
    // deberian vivir en device
    int centerIdx;
    int *outsideIdxs;

public :

    W2VEmbedding(Shape dict_shape, int context);
    Matrix W2V;     // es una matriz de (2*vocab_size, embed_size)
    
};