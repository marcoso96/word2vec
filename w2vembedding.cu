#include "w2vembedding.hh"

using namespace std;

// outOffset es por facilidad
// W2V tiene tamaño (2*vocab_size, embed_size)
W2VEmbedding::W2VEmbedding(Shape W2V_shape, int contextSize, int sents_num, int train_sents, int lr):
    cost(W2V_shape.y, contextSize, W2V_shape.x, lr), W2V(2*W2V_shape.x, W2V_shape.y)
{   
    W2V.allocateMemory();
    initW2V();

    this -> context = contextSize;  // tamaño de ventana. alrededor de una palabra central, se toman a lo sumo 2*context palabras
    this -> sents_num = sents_num;
    this -> train_sents = train_sents;
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
    float *deviceOutside = W2V.data_d.get()+(W2V.shape.x/2)*W2V.shape.y;

    curandCreateGenerator(&prgen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prgen, 12314ULL);

    curandGenerateNormal(prgen, W2V.data_d.get(), W2V.shape.x*W2V.shape.y, 0, 1.0f/W2V.shape.y);

    curandDestroyGenerator(prgen);

    // genero vocab_size*embed_size ceros
    cudaMemset(deviceOutside, 0, (W2V.shape.x/2)*W2V.shape.y*sizeof(float));
}
// centerIdx es la dirección de memoria que apunta al vector central

void W2VEmbedding::updateDictionary(int *h_Idx, int sentID, int cWordID, int low_bound, int up_bound)
{   
    int batch_size = (up_bound-low_bound);  // numero de elementos de contexto, ojo con las cuentas en direcciones de memoria
    assert(batch_size>0);
    // float* aux = (float *)malloc(sizeof(float)*batch_size);

    // copio los indices de las palabras contexto, a lo sumo 2*context
    cudaMemcpy(d_idx, &(h_Idx[sentID+low_bound]), (cWordID-low_bound)*sizeof(int), cudaMemcpyHostToDevice);          // index
    cudaMemcpy(d_idx, &(h_Idx[sentID+cWordID+1]), (up_bound - cWordID)*sizeof(int), cudaMemcpyHostToDevice);

    cost.lossAndGrad(W2V.data_d.get(), d_idx, h_Idx[sentID+cWordID], batch_size);
    cost.updateGradients(W2V.data_d.get(), )
}


