#include "w2vembedding.hh"

using namespace std;

// outOffset es por facilidad
// W2V tiene tamaño (2*vocab_size, embed_size)
W2VEmbedding::W2VEmbedding(Shape W2V_shape, int contextSize, int sents_num, int batch_size, double lr):
    cost(W2V_shape.y, W2V_shape.x, lr, batch_size), W2V(2*W2V_shape.x, W2V_shape.y)
{   
    W2V.allocateMemory();
    initW2V();

    this -> context = contextSize;  // tamaño de ventana. alrededor de una palabra central, se toman a lo sumo 2*context palabras
    this -> sents_num = sents_num;
    this -> outOffset = (W2V.shape.x/2)*W2V.shape.y;
    
    cudaMalloc(&d_idx, (2*context+1)*sizeof(int));
    cudaMemset(d_idx, 0, (2*context+1)*sizeof(int));
}

W2VEmbedding::~W2VEmbedding()
{
    cudaFree(d_idx);
}

void W2VEmbedding::initW2V()
{
    // genero vocab_size*embed_size aleatorios y vocab_size*embed_size seteados en ceros : Center y OutsideVectors
    curandGenerator_t prgen;
    double *deviceOutside = W2V.data_d.get()+(W2V.shape.x/2)*W2V.shape.y;

    curandCreateGenerator(&prgen, CURAND_RNG_PSEUDO_DEFAULT);
    gpuErrchk(cudaPeekAtLastError());
    curandSetPseudoRandomGeneratorSeed(prgen, time(NULL));
    gpuErrchk(cudaPeekAtLastError());
    curandGenerateNormalDouble(prgen, W2V.data_d.get(), W2V.shape.x*W2V.shape.y, 0.0, 1.0/W2V.shape.y) ;
    gpuErrchk(cudaPeekAtLastError());
    curandDestroyGenerator(prgen);
    gpuErrchk(cudaPeekAtLastError());

    // genero vocab_size*embed_size ceros
    cudaMemset(deviceOutside, 0, (W2V.shape.x/2)*W2V.shape.y*sizeof(double));
    gpuErrchk(cudaPeekAtLastError());

}
// centerIdx es la dirección de memoria que apunta al vector central

void W2VEmbedding::updateDictStep(int *h_Idx, int sentID, int cWordID, int low_bound, int up_bound)
{   
    int context_size = (up_bound-low_bound);  // numero de elementos de contexto, ojo con las cuentas en direcciones de memoria
    assert(context_size <= 2*context);
    assert(cWordID-low_bound >= 0);

    if(cWordID-low_bound > 0){
        // copio los indices de las palabras contexto, a lo sumo 2*context
        cudaMemcpy(d_idx, &(h_Idx[sentID+low_bound]), (cWordID-low_bound)*sizeof(int), cudaMemcpyHostToDevice);          // index
    }

    if(up_bound-cWordID > 0){
        cudaMemcpy(&d_idx[cWordID-low_bound], &(h_Idx[sentID+cWordID+1]), (up_bound - cWordID-1)*sizeof(int), cudaMemcpyHostToDevice);
    }

    cost.lossAndGrad(W2V.data_d.get(), d_idx, h_Idx[sentID+cWordID], context_size);
    cost.updateGradients(W2V.data_d.get(), h_Idx[sentID+cWordID]);
}

void W2VEmbedding::saveDict(string data_path)
{
    // print_matrix(W2V.data_d.get(), W2V.shape.x, W2V.shape.y);
    W2V.copyD2H();
    cnpy::npy_save(data_path, W2V.data_h.get(), {W2V.shape.x, W2V.shape.y}, "w");
}
