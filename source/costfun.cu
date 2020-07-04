#include "costfun.hh"

void print_matrix(double *Mat, int Mat_height, int Mat_width){

    double* aux = (double *)malloc(sizeof(double)*Mat_width*Mat_height);

    cudaMemcpy(aux, Mat, sizeof(double)*Mat_width*Mat_height, cudaMemcpyDeviceToHost);

    printf("fil : %d, col : %d\n", Mat_width, Mat_height);
    for(int i=0; i<Mat_height; i++)
    {
        for(int j=0; j<Mat_width; j++)
        {
            printf("%.10f ",aux[i*Mat_width+j]);
        }
        printf("\n");
    }

    free(aux);
}

struct max_exp
{
    double max;
    max_exp(double m){max = m;};
    __device__ double operator()(double y)
    {
        return exp(y-max);
    }
};

// non safe at all
__global__ void logitsSoftmax(double *wordVecs, double *Y_est, int centerIdx, int vocab_size, int embed_size, int offset)
{
    // para cada fila tomo los indices del thread 
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    double logits_value = 0.0;

    if (fil < vocab_size)
    {   
        for (int i=0 ; i < embed_size; i++)
        {   
            // recorro las filas de Offset vectors
            logits_value +=  wordVecs[offset+fil*embed_size+i]*wordVecs[centerIdx*embed_size+i];   
        }  
        Y_est[fil] = logits_value;
    }
}
// gradiente con respecto a la palabra clave (ya le paso el softmax)
// transpongo la matriz de palabras así le actualizo todo
__global__ void gradCenterVec(double* outsideVecs, double* Y_est, double *gradCenter,  int vocab_size, int embed_size)
{   
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    double grad = 0.0;

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
__global__ void gradOutsideVecs(double *centerVec, double *Y_est, double *gradOutside,  int vocab_size, int embed_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(fil < vocab_size)
    { 
        if (col < embed_size)
        {
            gradOutside[fil*embed_size+col] += Y_est[fil]*centerVec[col];   
        }
    }
    __syncthreads();
}

// update implica Y = Y_est - Y
__global__ void updateY(double *Y, double *loss, int* out_idxs, int currIdx, int batch_size)
{   
    // printf("%lf\t%lf\tind : %d \tsent ind : %d\n",Y[out_idxs[currIdx]], log(Y[out_idxs[currIdx]]), currIdx, out_idxs[currIdx]);
    *loss -= log(Y[out_idxs[currIdx]]);
    Y[out_idxs[currIdx]] += -1;
    __syncthreads();
}

__global__ void upCenter(double *centerVec, double *grad_center, double lr, int embed_size, int  batch_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(fil < embed_size)
    {   
        centerVec[fil] -= lr*grad_center[fil]/batch_size;   
    }
    __syncthreads();
}

__global__ void upOutside(double *outsideVecs, double *grad_outside, double lr, int embed_size, int vocab_size, int batch_size)
{   
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(fil < embed_size)
    {   
        if(col < vocab_size)
        {   
            outsideVecs[col*embed_size + fil] -= lr*grad_outside[col*embed_size + fil]/batch_size;   
        }
    }
    __syncthreads();
}

// le paso el vector central y los vectores outside 
// vec central es (embed_size, 1), vec outside es (k, embed_size)
// https://devblogs.nvidia.com/unified-memory-cuda-beginners/ por cudaMallocManaged
// vec central YA VIENE TRANSPUESTO, ver si es una decision piola o lo transpongo en kernel, c'est le meme
// agarro cada uno de los logits, los exponencio y obtengo una densidad de probabilidad
// de cada palabra externa dada una central
// cost es un vector de K elementos que me da una probabilidad empírica de lo cercanas que estan dos palabras en este espacio. es en el mismo sentido, la entropia conjunta entre la palabra real y_i {i=1,...,k}(con prob 1) y la palabra predicha y^{\hat}_i {i=1,...,k}

W2VCost::W2VCost(int embed_size, int vocab_size, double lr, int batch_size)
{   
    out_loss.open("out_loss_Cublas.txt");
    // el máximo que voy a requerir es context
    this -> embed_size = embed_size;
    this -> vocab_size = vocab_size;
    this -> out_offset = vocab_size*embed_size;
    this -> batch_size = batch_size;
    
    this -> lr = lr;
    this -> iteration = 0;

    cublasCreate(&(this -> handler));

    cudaMalloc(&Y_est, vocab_size*sizeof(double));
    cudaMalloc(&grad_center, embed_size*sizeof(double));    // (1, embed_size)
    cudaMalloc(&grad_outside, vocab_size*embed_size*sizeof(double));    // (context, embed_size)
    cudaMalloc(&loss, sizeof(double));

    cudaMemset(Y_est, 0, vocab_size*sizeof(double));
    cudaMemset(grad_center, 0,  embed_size*sizeof(double));
    cudaMemset(grad_outside, 0,  vocab_size*embed_size*sizeof(double));
    cudaMemset(loss, 0,  sizeof(double));
}

W2VCost::~W2VCost()
{   
    cublasDestroy(this -> handler);
    cudaFree(this -> grad_center);
    cudaFree(this -> grad_outside);
    cudaFree(this -> loss);
    cudaFree(this -> Y_est);
    out_loss.close();

}

// para cada palabra externa
void W2VCost::lossAndGrad(double* wordVecs, int* outsideIdxs,  int centerIdx, int context_size)
{      
    // double *aux = (double*)malloc(sizeof(double)*vocab_size*embed_size);
    // por cada palabra del contexto, actualizo
    for(int currentOutIdx=0; currentOutIdx<context_size; currentOutIdx++)
    {   
        W2VCost::softLoss(wordVecs, centerIdx);
        updateY<<<1,1>>>(Y_est, loss, outsideIdxs, currentOutIdx, batch_size);
        gpuErrchk(cudaPeekAtLastError());
        // // actualizo gradientes 
        W2VCost::gradCenter(&wordVecs[out_offset]);
        W2VCost::gradOutside(&wordVecs[centerIdx*embed_size]);
    }
}

void W2VCost::updateGradients(double* wordVecs, int centerIdx)
{   
    double loss_h;

    updateCenter(&wordVecs[embed_size*centerIdx]);
    cudaMemset(grad_center, 0,  embed_size*sizeof(double));
    gpuErrchk(cudaPeekAtLastError());

    updateOutside(&wordVecs[out_offset]);
    // print_matrix(grad_outside, vocab_size, embed_size);
    cudaMemset(grad_outside, 0,  vocab_size*embed_size*sizeof(double));
    gpuErrchk(cudaPeekAtLastError());

    this -> iteration ++;

    if((this -> iteration%PRINT_EVERY)== 0){
        
        if (exploss == 0)
        {  
            cudaMemcpy(&exploss, loss, sizeof(double), cudaMemcpyDeviceToHost);
            exploss /= batch_size;
        }
        else 
        {
            cudaMemcpy(&loss_h, loss, sizeof(double), cudaMemcpyDeviceToHost);
            loss_h /= batch_size;
            exploss = 0.95*exploss+0.05*loss_h;    
        }
        
        printf("Iter : %d\tLoss : %.10lf\n", iteration, exploss);
        out_loss << iteration << "," << exploss << endl;
    }
    // ACTUALIZO OUTSIDE CADA BATCH SIZE Y CAMBIO LR
    if((this -> iteration%this->batch_size) == 0) 
    { 
        cudaMemset(loss, 0, sizeof(double));
        this -> lr *= 0.5;
    }
}

void W2VCost::updateCenter(double* centerVec)
{   
        // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    upCenter<<<block_num, block_size>>>(centerVec, grad_center, lr, embed_size, batch_size);
    gpuErrchk(cudaPeekAtLastError());

}

void W2VCost::updateOutside(double* outsideVecs)
{   
    // necesito vocab_size threads
    dim3 block_size(8, 8);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x, (vocab_size+block_size.y-1)/block_size.y);

    upOutside<<<block_num, block_size>>>(outsideVecs, grad_outside, lr, embed_size, vocab_size, batch_size);
    gpuErrchk(cudaPeekAtLastError());
}

void W2VCost::softLoss(double *wordVecs, int centerVecIdx)
{   
    double sum = 0.0;
    double max;

    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((vocab_size+block_size.x-1)/block_size.x);

    assert(out_offset == vocab_size*embed_size);
    assert(centerVecIdx < vocab_size);

    #ifdef SIMPLECUDA
    logitsSoftmax<<<block_num, block_size>>>(wordVecs, Y_est, centerVecIdx, vocab_size, embed_size, out_offset);
    gpuErrchk(cudaPeekAtLastError());
    #endif 

    #ifdef CUBLAS
    stat = cublasDgemv(this->handler, CUBLAS_OP_T, embed_size, vocab_size, &alfa, &wordVecs[out_offset], embed_size, &wordVecs[embed_size*centerVecIdx], 1, &beta, Y_est, 1);
    #endif

    thrust::device_ptr<double>Y_dev = thrust::device_pointer_cast(Y_est);
    max = *(thrust::max_element(Y_dev, Y_dev+vocab_size));

    // para fomentar la estabilidad y las buenas costumbres
    thrust::transform(Y_dev, Y_dev+vocab_size, Y_dev, max_exp(max));
    sum = thrust::reduce(Y_dev, Y_dev+vocab_size, 0, thrust::plus<double>()); 

    thrust::transform(Y_dev, Y_dev+vocab_size, Y_dev, _1/sum);
    gpuErrchk(cudaPeekAtLastError());
}


void W2VCost::gradCenter(double *outsideVecs)
{   
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    #ifdef SIMPLECUDA
    gradCenterVec<<<block_num, block_size>>>(outsideVecs, Y_est, grad_center, vocab_size, embed_size);
    gpuErrchk(cudaPeekAtLastError());
    #endif

    #ifdef CUBLAS
    stat = cublasDgemv(this->handler, CUBLAS_OP_N, embed_size, vocab_size, &alfa, outsideVecs, embed_size, Y_est, 1, &beta, grad_center, 1);
    #endif
}

void W2VCost::gradOutside(double *centerVec)
{
    // cout <<"Batch size gradout : " << batch_size << endl;
    dim3 block_size(8, 8);
    dim3 block_num((vocab_size+block_size.x-1)/block_size.x, (embed_size+block_size.y-1)/block_size.y);
    
    gradOutsideVecs<<<block_num, block_size>>>(centerVec, Y_est, grad_outside, vocab_size, embed_size);
    gpuErrchk(cudaPeekAtLastError());
}