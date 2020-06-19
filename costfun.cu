#include "costfun.hh"

using namespace std;
using namespace thrust::placeholders;

void print_matrix(Matrix Mat){

    printf("fil : %lu, col : %lu\n", Mat.shape.x, Mat.shape.y);
    for(int i=0; i<Mat.shape.x; i++)
    {
        for(int j=0; j<Mat.shape.y; j++)
        {
            printf("%f ",Mat[i*Mat.shape.y+j]);
        }
        printf("\n");
    }
}

// super ineficiente
__global__ void logitsSoftmax(float *centerVec, float *outsideVecs, float *Y_est, int batch_size, int embed_size, int outIdx)
{
    // para cada fila tomo los indices del thread 
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    float logits_value = 0.0;

    if (fil < batch_size)
    {   
        for (int i=0 ; i < embed_size; i++)
        {   
            // recorro las filas de O
            logits_value +=  outsideVecs[fil*embed_size+i]*centerVec[i];
            
        }
        
        Y_est[fil] = expf(logits_value);
        
        // printf("%f\t", Y_est[fil]);

        // if (fil == outIdx) Y_est[fil] -= 1;

        
    }
}

// // sum : \sum_{i=1}^{logits_size}exp(logits[i]), softmax/cost tiene la misma dimensión que logits
// __global__ void totalSoftmax(float *logits, float sum, int outside_window_size)
// {
//     int fil = blockIdx.x * blockDim.x + threadIdx.x;

//     if (fil<outside_window_size)
//     {
//         logits[fil] /= sum;
//     }
// }

// gradiente con respecto a la palabra clave (ya le paso el softmax)
// transpongo la matriz de palabras así le actualizo todo
// deprecated
__global__ void gradCenterVec(float* outsideVecs, float* Y_est, float *gradCenter,  int outside_window_size, int embed_size)
{   
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    float grad = 0.0;

    if (fil<embed_size)
    {
        for (int i=0 ; i < outside_window_size; i++)
        {
            grad += outsideVecs[i*embed_size+fil]*Y_est[i];
        }

        gradCenter[fil] += grad;
    }
}

// hago producto externo entre center vecs y y-y_est para actualizar palabras outside
__global__ void gradOutsideVecs(float *centerVec, float *Y_est, float *gradOutside,  int outside_window_size, int embed_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < embed_size)
    {
        if(fil < outside_window_size)
        {   
            gradOutside[fil*embed_size+col] += Y_est[fil]*centerVec[col];   
        }
    }
}

// update implica Y = Y_est - Y
__global__ void updateY(float *Y, float *loss, int idx)
{   
    Y[idx] += -1;
    *loss += logf(Y[idx]);
}

// le paso el vector central y los vectores outside 
// vec central es (embed_size, 1), vec outside es (k, embed_size)
// https://devblogs.nvidia.com/unified-memory-cuda-beginners/ por cudaMallocManaged
// vec central YA VIENE TRANSPUESTO, ver si es una decision piola o lo transpongo en kernel, c'est le meme
// agarro cada uno de los logits, los exponencio y obtengo una densidad de probabilidad
// de cada palabra externa dada una central
// cost es un vector de K elementos que me da una probabilidad empírica de lo cercanas que estan dos palabras en este espacio. es en el mismo sentido, la entropia conjunta entre la palabra real y_i {i=1,...,k}(con prob 1) y la palabra predicha y^{\hat}_i {i=1,...,k}

W2VCost::W2VCost(int embed_size, int context)
{
    // el máximo que voy a requerir es context
    
    this -> context = context;
    this -> embed_size = embed_size;

    cudaMalloc(&Y_est, 2*context*sizeof(float));
    cudaMalloc(&grad_center, embed_size*sizeof(float));    // (1, embed_size)
    cudaMalloc(&grad_outside, 2*context*embed_size*sizeof(float));    // (context, embed_size)
    cudaMalloc(&loss, sizeof(float));

    cudaMemset(Y_est, 0, 2*context*sizeof(float));
    cudaMemset(grad_center, 0,  embed_size*sizeof(float));
    cudaMemset(grad_outside, 0,  2*context*embed_size*sizeof(float));
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
void W2VCost::lossAndGrad(float *centerVec, float *outsideVecs, int batch_size)
{   
    this -> batch_size = batch_size;    // tamaño de contexto en oracion actual
    // rompe con el paralelismo pero lo veré después
    cout << "Batch size : " << (this -> batch_size) <<endl;
    
    for(int outsideIdx=0; outsideIdx<(this->batch_size); outsideIdx++)
    {   
        cout << cudaMemset(Y_est, 0, (2*context)*sizeof(float)) << endl;
        W2VCost::softLoss(outsideVecs, centerVec, outsideIdx);  
        // updateY<<<1,1>>>(Y_est, loss, outsideIdx);

        // // actualizo gradientes 
        // W2VCost::gradCenter(outsideVecs);
        // W2VCost::gradOutside(centerVec);
    }
    
    // sobreescribo estos vectores 
    cudaMemcpy(outsideVecs, grad_outside, (this->batch_size)*embed_size*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(centerVec, grad_center, embed_size*sizeof(float), cudaMemcpyDeviceToDevice);
}
// centerVec : (embed_size, 1)
// outsideVecIdx : int
// outsideVecs : (window_size, embed_size)
// check
void W2VCost::softLoss(float *outsideVecs, float *centerVec, int outIdx)
{   
    float sum = 0.0;

    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((batch_size+block_size.x-1)/block_size.x);

    //hago los k productos punto entre central y las outside
    // logitsSoftmax<<<block_num, block_size>>>(centerVec, outsideVecs, Y_est, batch_size, embed_size, outIdx);

    sum = thrust::reduce(thrust::device_ptr<float>(Y_est), thrust::device_ptr<float>(Y_est+ batch_size)); 
    // // // acá realmente hago softmax
    // thrust::transform(logits_ptr, logits_ptr+batch_size, logits_ptr, _1/sum);

    // *loss = -logf(*loss); 
}

// check
void W2VCost::gradCenter(float *outsideVecs)
{   
    // cout <<"Batch size gradcen : " << batch_size << endl;
    // necesito embed_size threads
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    gradCenterVec<<<block_num, block_size>>>(outsideVecs, Y_est, grad_center, batch_size, embed_size);
}

void W2VCost::gradOutside(float *centerVec)
{
    // cout <<"Batch size gradout : " << batch_size << endl;
    dim3 block_size(8, 8);
    dim3 block_num((batch_size+block_size.x-1)/block_size.x, (embed_size+block_size.y-1)/block_size.y);
    
    gradOutsideVecs<<<block_num, block_size>>>(centerVec, Y_est, grad_outside, batch_size, embed_size);
}