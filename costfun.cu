#include "costfun.hh"

using namespace std;


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

struct exp_functor 
{
    __device__
        float operator()(float x) const     
        { 
            return expf(x); 
        }
};
// hago un solo producto entre outside y center, brutalmente ineficiente será? Lo veremos pronto
// calculo U^T*v_c (U : (outside_window, embed_size), u_o : (1, embed_size), v_c : (1, embed_size))
// le paso
__global__ void logitsSoftmax(float *centerVec, float *outsideVecs, float *logits, int outside_window_size, int embed_size)
{
    // para cada fila tomo los indices del thread 
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    float logits_value = 0.0;

    if (fil < outside_window_size)
    {   
        for (int i=0 ; i < embed_size; i++)
        {
            // recorro las filas de O
            logits_value +=  outsideVecs[fil*embed_size+i]*centerVec[i];
        }

        
        logits[fil] = logits_value;
    }
}

// sum : \sum_{i=1}^{logits_size}exp(logits[i]), softmax/cost tiene la misma dimensión que logits
__global__ void totalSoftmax(float *logits, float *cost, float sum, int outside_window_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;

    if (fil<outside_window_size)
    {
        cost[fil] = logits[fil]/(sum);
    }
}

// gradiente con respecto a la palabra clave (ya le paso el softmax)
// transpongo la matriz de palabras así le actualizo todo
// deprecated
__global__ void gradCenterVec(float * outsideVecs, float *Y_est, float *gradCenter,  int outside_window_size, int embed_size, int outsideVecIdx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float grad = 0.0;

    if (col<embed_size)
    {
        for (int i=0 ; i < outside_window_size; i++)
        {
            // O^T * (y_est-y) donde y es 1 si es el outside vec y 0 otherwise
            if (i==outsideVecIdx)
            {
                grad += outsideVecs[i*embed_size+col]*(Y_est[i]-1);

            }
            else
            {
                grad += outsideVecs[i*embed_size+col]*(Y_est[i]);
            }
            
            printf("%f\n", Y_est[i]);
        
        }

        gradCenter[col] = grad;
    }
}

// hago producto externo entre center vecs y y-y_est para actualizar palabras outside
__global__ void gradOutsideVecs(float *centerVec, float *Y_est, float *gradOutside,  int outside_window_size, int embed_size)
{
    int fil = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if (col<embed_size)
    {
        if(fil< outside_window_size)
        {   
            gradOutside[fil*embed_size+col] = Y_est[fil]*centerVec[col];
        }
    }
}

// update implica Y = Y_est - Y
__global__ void updateY(float *Y, int idx)
{
    Y[idx] += -1;
}
// le paso el vector central y los vectores outside 
// vec central es (embed_size, 1), vec outside es (k, embed_size)
// https://devblogs.nvidia.com/unified-memory-cuda-beginners/ por cudaMallocManaged
// vec central YA VIENE TRANSPUESTO, ver si es una decision piola o lo transpongo en kernel, c'est le meme
// agarro cada uno de los logits, los exponencio y obtengo una densidad de probabilidad
// de cada palabra externa dada una central
// cost es un vector de K elementos que me da una probabilidad empírica de lo cercanas que estan dos palabras en este espacio. es en el mismo sentido, la entropia conjunta entre la palabra real y_i {i=1,...,k}(con prob 1) y la palabra predicha y^{\hat}_i {i=1,...,k}

// para cada palabra externa
void W2VCost::lossAndGrad(Matrix &centerVec, Matrix &outsideVecs, int outsideVecIdx, float *loss, Matrix &gradCenter, Matrix &gradOutside)
{
    // chequeo que tengan igual embed_size
    assert(outsideVecs.shape.y == centerVec.shape.x );

    Matrix Y_est(outsideVecs.shape.x, 1);        // k valores = cantidad de palabras

    // copio memoria en la matriz de costo en device
    Y_est.allocateMemory();

    W2VCost::softLoss(outsideVecs, centerVec, Y_est, loss, outsideVecIdx);  

    // debo operar sobre Y_est tal que Y_est = Y_est - 1[outsideVecIdx]
    updateY<<<1, 1>>>(Y_est.data_d.get(), outsideVecIdx);

    W2VCost::gradCenter(outsideVecs, Y_est, gradCenter);
    W2VCost::gradOutside(centerVec, Y_est, gradOutside);


}

// centerVec : (embed_size, 1)
// outsideVecIdx : int
// outsideVecs : (window_size, embed_size)
// check
void W2VCost::softLoss(Matrix &outsideVecs, Matrix &centerVec, Matrix& Y_est, float *loss, int outsideVecIdx)
{   
    float sum = 0;
    float *logits;

    // variables locales en device 
    cudaMalloc(&logits, outsideVecs.shape.x*sizeof(float));

    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((outsideVecs.shape.x+block_size.x-1)/block_size.x);

    //hago los k productos punto entre central y las outside
    logitsSoftmax<<<block_size, block_num>>>(centerVec.data_d.get(), outsideVecs.data_d.get(), logits, outsideVecs.shape.x, outsideVecs.shape.y);

    // hago en thrust porque es piola
    //transformo a exponencial
    // reduzco a suma
    thrust::device_ptr<float> logits_ptr = thrust::device_pointer_cast(logits);
    thrust::transform(logits_ptr, logits_ptr+outsideVecs.shape.x, logits_ptr, exp_functor());
    sum = thrust::reduce(logits_ptr,logits_ptr+outsideVecs.shape.x, 0, thrust::plus<float>()); 
    
    // acá realmente hago softmax
    totalSoftmax<<<block_size, block_num>>>(logits, Y_est.data_d.get(), sum, outsideVecs.shape.x);
    // libero memoria en device
    cudaFree(logits);

    cudaMemcpy(loss, &Y_est.data_d.get()[outsideVecIdx], sizeof(float), cudaMemcpyDeviceToHost);
    *loss = -logf(*loss); 
}


// check
void W2VCost::gradCenter(Matrix &outsideVecs, Matrix &Y_est, Matrix &gradCenter)
{

    // // necesito embed_size threads
    // dim3 block_size(256);
    // dim3 block_num((outsideVecs.shape.y+block_size.x-1)/block_size.x);

    // gradCenterVec<<<block_size, block_num>>>(outsideVecs.data_d.get(), Y_est.data_d.get(), gradCenter.data_d.get(), outsideVecs.shape.x, outsideVecs.shape.y, outsideVecIdx);
    cublasHandle_t handle;
    const float alfa = 1.0;
    const float beta = 0.0;

    cublasCreate(&handle);
    // la traspuse a outsideVecs
    cublasSgemv(handle, CUBLAS_OP_T, outsideVecs.shape.y, outsideVecs.shape.x, &alfa, outsideVecs.data_d.get(), outsideVecs.shape.y, Y_est.data_d.get(), 1, &beta, gradCenter.data_d.get(), 1);

}


void W2VCost::gradOutside(Matrix &centerVec, Matrix &Y_est, Matrix &gradOutside)
{

    dim3 block_size(256, 256);
    dim3 block_num((Y_est.shape.x+block_size.x-1)/block_size.x, (centerVec.shape.y+block_size.y-1)/block_size.y);
    
    gradOutsideVecs<<<block_size, block_num>>>(centerVec.data_d.get(), Y_est.data_d.get(), gradOutside.data_d.get(), gradOutside.shape.x, gradOutside.shape.y);
}