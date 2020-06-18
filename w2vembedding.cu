#include "w2vembedding.hh"
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

// outOffset es por facilidad
__global__ void updateGrads(float* dictionary, float* grad_center, float *grad_outside, int *d_idx, int embed_size, int outOffset, int low_bound, int batch_size, int sents_num)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < embed_size)
    {
        if(fil < batch_size)
        {

            if(fil != low_bound) 
            {
                dictionary[(d_idx[fil]*embed_size+col)+outOffset] +=  grad_outside[fil*embed_size+col]/sents_num;
            }

            else                    // caso center - grad_center ~ (1, embed_size)
            {
                dictionary[d_idx[fil]*embed_size+col+outOffset] += grad_center[col]/sents_num;
            

            }

            
        }
    }
}

W2VEmbedding::W2VEmbedding(Shape W2V_shape, int contextSize, int sents_num):
    cost(W2V_shape.y, contextSize), W2V(2*W2V_shape.x, W2V_shape.y)
{   
    W2V.allocateMemory();
    initW2V();

    this -> context = contextSize;  // tamaño de ventana. alrededor de una palabra central, se toman a lo sumo 2*context palabras
    this -> sents_num = sents_num;
    this -> outOffset = (W2V.shape.x/2)*W2V.shape.y;
    
    cudaMalloc(&d_centerVec, W2V.shape.y*sizeof(float));    // (1, embed_size)
    cudaMalloc(&d_outsideVecs, 2*context*W2V.shape.y*sizeof(float));    // (context, embed_size)
    cudaMalloc(&d_idx, 2*context*sizeof(int));

    cudaMemset(d_centerVec, 0, W2V.shape.y*sizeof(float));
    cudaMemset(d_outsideVecs, 0, 2*context*W2V.shape.y*sizeof(float));
    cudaMemset(d_idx, 0, 2*context*sizeof(int));

}

W2VEmbedding::~W2VEmbedding()
{
    cudaFree(d_centerVec);
    cudaFree(d_outsideVecs);
    cudaFree(d_idx);
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
// centerIdx es la dirección de memoria que apunta al vector central

void W2VEmbedding::updateDictionary(int *h_Idx, int sentID, int cWordID, int low_bound, int up_bound)
{   
    
    int batch_size = (up_bound-low_bound);
    int j = 0;
    assert(batch_size>0);

    // for(int i=0; i<up_bound; i++)
    // {
    //     cout<< centerIdx[i] << '\n';
    // }
    // vector de posiciones en diccionario - copio lo que obtengo para una palabra -- esto podría ser modificado
    cudaMemcpy(d_idx, &h_Idx[sentID+low_bound], batch_size, cudaMemcpyHostToDevice);

    // copio memoria del vector central
    cudaMemcpy(d_centerVec, &W2V.data_d.get()[W2V.shape.y*h_Idx[sentID+cWordID]], W2V.shape.y*sizeof(float), cudaMemcpyDeviceToDevice);

    // Mapeo a vectores
    for(int idx = low_bound; idx <= up_bound; idx++)
    {
        if (idx == cWordID) continue;  // la central la omito
        
        // agarro el vector correspondiente a cada palabra en outsideVectors y la voy copiando

        cudaMemcpy(&d_outsideVecs[j*W2V.shape.y], &W2V.data_d.get()[outOffset+h_Idx[sentID + idx]*W2V.shape.y], W2V.shape.y*sizeof(float), cudaMemcpyDeviceToDevice);
        j++;
    }

    // Hasta acá, chequeado
    
    // calculo las actualizaciones para la palabra central y las de los costados
    // sobre d_centerVec y d_outsideVecs 
    cost.lossAndGrad(d_centerVec, d_outsideVecs, batch_size);
    
    // kernet bidimensional para actualizar matriz
    dim3 block_size(256, 256);
    dim3 block_num((batch_size+block_size.x-1)/block_size.x, (W2V.shape.y+block_size.y-1)/block_size.y);

    updateGrads<<<block_num, block_size>>>(W2V.data_d.get(), d_centerVec, d_outsideVecs, d_idx, W2V.shape.y, outOffset, low_bound, batch_size, sents_num);
}