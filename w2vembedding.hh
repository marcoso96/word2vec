#include "matrix.hh"
#include <iostream>
#include <curand.h>
#include <cuda.h>
#include "costfun.hh"
#include <iostream>


class W2VEmbedding{
private :

    W2VCost cost;
    Matrix W2V;                 // es una matriz de (2*vocab_size, embed_size)
    
    int outOffset;
    int context;
    
    int* d_idx;                 // acá voy a poner los indices en device
    int sents_num;              // número de oraciones para entrenamiento
    int train_sents;
    
    void initW2V();

public :

    W2VEmbedding(Shape dict_shape, int context, int sents_num, int train_sents, int lr);
    ~W2VEmbedding();
    
    // cargo las palabras de entrada
    void updateDictionary(int *h_Idx, int sentID, int cWordID, int low_bound, int up_bound);
};