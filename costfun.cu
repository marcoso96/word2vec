#include "costfun.hh"

using namespace std;
using namespace thrust::placeholders;

void print_matrix(float *Mat, int Mat_height, int Mat_width){

    printf("fil : %d, col : %d\n", Mat_width, Mat_height);
    for(int i=0; i<Mat_height; i++)
    {
        for(int j=0; j<Mat_width; j++)
        {
            printf("%f ",Mat[i*Mat_width+j]);
        }
        printf("\n");
    }
}

__global__ void upCenter(float *centerVec, float *grad_center, int lr, int embed_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(fil < embed_size)
    {   
        centerVec[fil] -= lr*grad_center[fil];   
    }
}

__global__ void upOutside(float *outsideVec, float *grad_outside, int lr, int embed_size, int vocab_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(fil < embed_size)
    {   
        if(col < vocab_size)
        {   
            outsideVec[fil*embed_size + col] -= lr*grad_outside[fil*embed_size + col];   
        }
    }
}

__global__ void logitsSoftmax(float *wordVecs, float *Y_est, int centerIdx, int vocab_size, int embed_size)
{
    // para cada fila tomo los indices del thread 
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    float logits_value = 0.0;

    if (fil < vocab_size)
    {   
        int offset = embed_size*vocab_size;

        for (int i=0 ; i < embed_size; i++)
        {   
            // recorro las filas de Offset vectors
            logits_value +=  wordVecs[offset+fil*embed_size+i]*wordVecs[centerIdx*embed_size+i];   
        }

        Y_est[fil] = expf(logits_value);
 
    }
}
// gradiente con respecto a la palabra clave (ya le paso el softmax)
// transpongo la matriz de palabras así le actualizo todo
// deprecated
__global__ void gradCenterVec(float* outsideVecs, float* Y_est, float *gradCenter,  int vocab_size, int embed_size)
{   
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    float grad = 0.0;

    if (fil<embed_size)
    {
        for (int i=0 ; i < vocab_size; i++)
        {
            grad += outsideVecs[i*embed_size+fil]*Y_est[i];
        }

        gradCenter[fil] += grad;
    }
    __syncthreads();
}

// hago producto externo entre center vecs y y-y_est para actualizar palabras outside
__global__ void gradOutsideVecs(float *centerVec, float *Y_est, float *gradOutside,  int vocab_size, int embed_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < embed_size)
    {
        if(fil < vocab_size)
        {   
            gradOutside[fil*embed_size+col] += Y_est[fil]*centerVec[col];   
        }
    }
    __syncthreads();
}

// update implica Y = Y_est - Y
__global__ void updateY(float *Y, float *loss, int* out_idxs, int currIdx)
{   
    Y[out_idxs[currIdx]] += -1;
    *loss += logf(Y[out_idxs[currIdx]]);
}

// le paso el vector central y los vectores outside 
// vec central es (embed_size, 1), vec outside es (k, embed_size)
// https://devblogs.nvidia.com/unified-memory-cuda-beginners/ por cudaMallocManaged
// vec central YA VIENE TRANSPUESTO, ver si es una decision piola o lo transpongo en kernel, c'est le meme
// agarro cada uno de los logits, los exponencio y obtengo una densidad de probabilidad
// de cada palabra externa dada una central
// cost es un vector de K elementos que me da una probabilidad empírica de lo cercanas que estan dos palabras en este espacio. es en el mismo sentido, la entropia conjunta entre la palabra real y_i {i=1,...,k}(con prob 1) y la palabra predicha y^{\hat}_i {i=1,...,k}

W2VCost::W2VCost(int embed_size, int context, int vocab_size, int lr)
{
    // el máximo que voy a requerir es context
    this -> context = context;
    this -> embed_size = embed_size;
    this -> vocab_size = vocab_size;
    this -> out_offset = vocab_size*embed_size;
    this -> lr = lr;
    this -> iteration = 0;

    cudaMalloc(&Y_est, vocab_size*sizeof(float));
    cudaMalloc(&grad_center, embed_size*sizeof(float));    // (1, embed_size)
    cudaMalloc(&grad_outside, vocab_size*embed_size*sizeof(float));    // (context, embed_size)
    cudaMalloc(&loss, sizeof(float));

    cudaMemset(Y_est, 0, vocab_size*sizeof(float));
    cudaMemset(grad_center, 0,  embed_size*sizeof(float));
    cudaMemset(grad_outside, 0,  vocab_size*embed_size*sizeof(float));
    cudaMemset(loss, 0,  sizeof(float));
}

W2VCost::~W2VCost()
{
    cudaFree(this -> grad_center);
    cudaFree(this -> grad_outside);
    cudaFree(this -> loss);
    cudaFree(this -> Y_est);
}

// para cada palabra externa
void W2VCost::lossAndGrad(float* wordVecs, int* outsideIdxs,  int centerIdx, int batch_size)
{      
    float *aux = (float*)malloc(sizeof(float)*vocab_size*embed_size);
    
    cout<< "Batch size\n" << batch_size <<endl;
    // por cada palabra del contexto, actualizo
    for(int currentOutIdx=0; currentOutIdx<batch_size; currentOutIdx++)
    {   
        W2VCost::softLoss(wordVecs, centerIdx);  
        updateY<<<1,1>>>(Y_est, loss, outsideIdxs, currentOutIdx);

        // // actualizo gradientes 
        W2VCost::gradCenter(&wordVecs[out_offset]);
        W2VCost::gradOutside(&wordVecs[centerIdx*embed_size]);

    }
    
    cudaMemcpy(aux, grad_outside, vocab_size*embed_size*sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix(aux, vocab_size, embed_size);

    free(aux);

    // ACÁ HACER UPDATE DE LA MATRIZ DE OUTPUT CONSIDERANDO EL GRAD DESCENT >> PASARLO o UPDATEAR ACÁ

    // cudaMemset(grad_center, 0,  embed_size*sizeof(float));
    // cudaMemset(grad_outside, 0,  vocab_size*embed_size*sizeof(float));   
}

void W2V::updateGradients(float* wordVecs, int centerIdx)
{
    
    updateCenter(&wordVecs[embed_size*centerIdx]);
    updateOutside(&wordVecs[out_offset]);
    
    this -> lr *= 0.5;
    this -> iteration ++;
}

void W2VCost::updateCenter(float* centerVec)
{
        // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    upCenter<<<block_num, block_size>>>(centerVec, grad_center, lr, embed_size);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(grad_center, 0,  embed_size*sizeof(float));
    gpuErrchk(cudaPeekAtLastError());
}
void W2VCost::updateOutside(float* outsideVecs)
{
    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((vocab_size+block_size.x-1)/block_size.x, (embed_size+block_size.y-1)/block_size.y);

    upOutside<<<block_num, block_size>>>(outsideVecs, grad_outside, lr, embed_size,  vocab_size);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(grad_outside, 0,  vocab_size*embed_size*sizeof(float));
    gpuErrchk(cudaPeekAtLastError());
}


// centerVec : (embed_size, 1)
// outsideVecIdx : int
// outsideVecs : (window_size, embed_size)
// check
void W2VCost::softLoss(float *wordVecs, int centerVecIdx)
{   
    float sum = 0.0;
    
    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((vocab_size+block_size.x-1)/block_size.x);

    // hago los k productos punto entre central y las outside
    logitsSoftmax<<<block_num, block_size>>>(wordVecs, Y_est, centerVecIdx, vocab_size, embed_size);

    thrust::device_ptr<float>Y_dev = thrust::device_pointer_cast(Y_est);
    sum = thrust::reduce(Y_dev, Y_dev+vocab_size, 0, thrust::plus<float>()); 
    // // acá realmente hago softmax
    thrust::transform(Y_dev, Y_dev+vocab_size, Y_dev, _1/sum);
    gpuErrchk(cudaPeekAtLastError());
    // *loss = -logf(*loss); 
}

// check
void W2VCost::gradCenter(float *outsideVecs)
{   
    // cout <<"Batch size gradcen : " << batch_size << endl;
    // necesito embed_size threads
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    gradCenterVec<<<block_num, block_size>>>(outsideVecs, Y_est, grad_center, vocab_size, embed_size);
    gpuErrchk(cudaPeekAtLastError());
}

void W2VCost::gradOutside(float *centerVec)
{
    // cout <<"Batch size gradout : " << batch_size << endl;
    dim3 block_size(8, 8);
    dim3 block_num((vocab_size+block_size.x-1)/block_size.x, (embed_size+block_size.y-1)/block_size.y);
    
    gradOutsideVecs<<<block_num, block_size>>>(centerVec, Y_est, grad_outside, vocab_size, embed_size);
    gpuErrchk(cudaPeekAtLastError());
}