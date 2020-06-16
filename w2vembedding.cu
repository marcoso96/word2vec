#include "w2vembedding.hh"
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

W2VEmbedding::W2VEmbedding(Shape W2V_shape):
    W2V(2*W2V_shape.x, W2V_shape.y)
{
    W2V.allocateMemory();
    initW2V();
}

void W2VEmbedding::initW2V()
{
    // genero vocab_size*embed_size aleatorios y vocab_size*embed_size seteados en ceros : Center y OutsideVectors
    curandGenerator_t prgen;
    float *deviceOutside = W2V.data_d.get()+W2V.shape.x/2*W2V.shape.y;

    cout << "Create Generator : " << curandCreateGenerator(&prgen, CURAND_RNG_PSEUDO_DEFAULT)<< endl;
    cout << "Seed : " << curandSetPseudoRandomGeneratorSeed(prgen, 12314ULL) << endl;

    cout << "Generate Normal : " << curandGenerateNormal(prgen, W2V.data_d.get(), W2V.shape.x*W2V.shape.y, 0, 1.0f/W2V.shape.y)<< endl;

    cout << "Destroy : " << curandDestroyGenerator(prgen)<< endl;

    // genero vocab_size*embed_size ceros
    cout << "Memset : " << cudaMemset(deviceOutside, 0, W2V.shape.x/2*W2V.shape.y*sizeof(float)) << endl;
}