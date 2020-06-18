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
__global__ void logitsSoftmax(float *centerVec, float *outsideVecs, float *Y_est, int batch_size, int embed_size)
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
        printf("%f\n", Y_est[fil]);
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
            grad += outsideVecs[i*embed_size+fil]*(Y_est[i]);
        }

        gradCenter[fil] += grad;
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

    cout << "alloc " << cudaMalloc(&Y_est, 2*context*sizeof(float))<< endl;
    cout << "alloc " << cudaMalloc(&grad_center, embed_size*sizeof(float))<< endl;    // (1, embed_size)
    cout << "alloc " <<cudaMalloc(&grad_outside, 2*context*embed_size*sizeof(float))<< endl;    // (context, embed_size)
    cout << "alloc " << cudaMalloc(&loss, sizeof(float))<< endl;

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
}

// para cada palabra externa
void W2VCost::lossAndGrad(float *centerVec, float *outsideVecs, int batch_size)
{   
    this -> batch_size = batch_size;    // tamaño de contexto en oracion actual
    // rompe con el paralelismo pero lo veré después
    cout << "Batch size : " << batch_size <<endl;
    cout << "Context : " << context << endl;
    float* aux = (float *)malloc(sizeof(float)*embed_size);

    for(int outsideIdx=0; outsideIdx<batch_size; outsideIdx++)
    {   
        cudaMemset(Y_est, 0, 2*context*sizeof(float));
        
        // PRUEBA
        cudaMemcpy(aux, &outsideVecs[outsideIdx*embed_size], sizeof(float)*embed_size, cudaMemcpyDeviceToHost);
        for(int i=0; i<embed_size ;i++)
        {
            cout << aux[i] << "\t";
        }
        cout << endl;
        // END PRUEBA

        W2VCost::softLoss(outsideVecs, centerVec);  
    
        // debo operar sobre Y_est tal que Y_est = Y_est - 1[outsideVecIdx]
        // updateY<<<1, 1>>>(Y_est, loss, outsideIdx);
        
        // // actualizo gradientes 
        // W2VCost::gradCenter(outsideVecs);
        // W2VCost::gradOutside(centerVec);
    }

    free(aux);
    // // sobreescribo estos vectores 
    // cudaMemcpy(outsideVecs, grad_outside, batch_size*embed_size*sizeof(float), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(centerVec, grad_center, embed_size*sizeof(float), cudaMemcpyDeviceToDevice);
}
// centerVec : (embed_size, 1)
// outsideVecIdx : int
// outsideVecs : (window_size, embed_size)
// check
void W2VCost::softLoss(float *outsideVecs, float *centerVec)
{   
    float sum = 0.0;
    float *h_print = (float *)malloc(sizeof(float)*batch_size);

    // necesito vocab_size threads
    dim3 block_size(256);
    dim3 block_num((batch_size+block_size.x-1)/block_size.x);

    // //hago los k productos punto entre central y las outside
    logitsSoftmax<<<block_num, block_size>>>(centerVec, outsideVecs, Y_est, batch_size, embed_size);

    cudaMemcpy(h_print, Y_est, batch_size*sizeof(float), cudaMemcpyDeviceToHost);


    // for(int i=0; i<batch_size; i++)
    // {
    //     cout << h_print[i] << '\t';
    // }
    // cout << endl;
    // thrust::device_ptr<float> logits_ptr = thrust::device_pointer_cast(Y_est);
    // sum = thrust::reduce(logits_ptr,logits_ptr+batch_size); 
    
    
    // // acá realmente hago softmax
    // thrust::transform(logits_ptr, logits_ptr+batch_size, logits_ptr, _1/sum);

    // cudaMemcpy(h_print, Y_est, sizeof(float)*batch_size, cudaMemcpyDeviceToHost);

    // for(int i=0; i<batch_size; i++)
    // {
    //     cout << h_print[i] << endl;
    // }
    // free(h_print);



    // *loss = -logf(*loss); 
}

// check
void W2VCost::gradCenter(float *outsideVecs)
{
    // necesito embed_size threads
    dim3 block_size(256);
    dim3 block_num((embed_size+block_size.x-1)/block_size.x);

    gradCenterVec<<<block_num, block_size>>>(outsideVecs, Y_est, grad_center, batch_size, embed_size);

}


void W2VCost::gradOutside(float *centerVec)
{
    dim3 block_size(256, 256);
    dim3 block_num((batch_size+block_size.x-1)/block_size.x, (embed_size+block_size.y-1)/block_size.y);
    
    gradOutsideVecs<<<block_num, block_size>>>(centerVec, Y_est, grad_outside, batch_size, embed_size);
}