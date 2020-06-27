#include "matrix.hh"
#include "layers.hh"


// Testeado, ponele
void Matrix::print_matrix(){

    copyD2H();

    printf("fil : %lu, col : %lu\n", shape.x, shape.y);
    
    for(int i=0; i< shape.x; i++)
    {
        for(int j=0; j< shape.y; j++)
        {
            printf("%f ", this->data_h.get()[i*shape.y+j]);
        }
        printf("\n");
    }
}
// ---------- SHAPE -----------
// constructor de Shape para que no le ponga mierda el compilador
// dado que size_t es un objeto en Shape, puedo inicializarlo así
Shape::Shape(size_t x, size_t y) : 
    x(x), y(y) 
    { }

// ---------- MATRIX -----------

// funciones del constructor
// a la matriz le pongo todos los atributos vacios
Matrix::Matrix(size_t x_dim, size_t y_dim) :
    shape(x_dim, y_dim), data_d(nullptr), data_h(nullptr),
    d_allocated(false), h_allocated(false)
    {  }

Matrix::Matrix(Shape shape) :
    Matrix(shape.x, shape.y)
    {  }

// métodos de alocacion de memoria en host y device
void Matrix::allocateMemory() 
{
    allocateHostMemory();
    allocateDevMemory();
}

// aloco memoria en caso de no tenerla
void Matrix::allocateMemoryIfNAll(Shape shape) 
{   
    // chequeo si estan alocadas en memoria estas cosas y sino, le pongo el shape y pimba
    if (!d_allocated && !h_allocated)
    {

        this -> shape = shape;
        allocateMemory();
    }
}
// método para alocación de matriz en mem host
void Matrix::allocateHostMemory() 
{
    if (!h_allocated) {
        
        // le calzo la memoria en host
        // [&] captura todas las variables en scope por referencia / lambda function que deletea la memoria apuntada cuando tenga que realizar la morición de la matrícula
        data_h = std::shared_ptr<double>(new double[shape.x * shape.y], [&](double * ptr){delete[] ptr; }); 
        memset(data_h.get(), 0, shape.x * shape.y * sizeof(double));
        // le aviso que ya aloque memoria en host
        h_allocated = true;
    }
}

void Matrix::allocateDevMemory()
{

    if (!d_allocated)
    {
        double * device_memory = nullptr;
        
        // aloco memoria en GPU y verifico que onda si hay errores
        cudaMalloc(&device_memory, shape.x*shape.y*sizeof(double));
        NNExc::thIfDevErr("No se puede alocar memoria para tensor");

        cudaMemset(device_memory, 0, shape.x*shape.y*sizeof(double));
        // de vuelta, le paso al puntero inteligente el método de destucción
        data_d = std::shared_ptr<double>(device_memory,
                                        [&](double *ptr){ cudaFree(ptr); });

        d_allocated = true;
    }
}

// método de copiado de memoria de host a device
void Matrix::copyH2D() {
    
    // chequeo que esten alocadas las memorias en host y device
    if (d_allocated && h_allocated) {
        
        cudaMemcpy(data_d.get(), data_h.get(), shape.x*shape.y*sizeof(double),cudaMemcpyHostToDevice);
        NNExc::thIfDevErr("No se puede copiar los datos de host a device\n");
    }  
    else {
        // no hay memoria alocada, hago un throw
        NNExc::thIfDevErr("No se puede copiar los datos porque la memoria no se halla alocada : H2D");
    }
}
// método de copiado de memoria de device a host
void Matrix::copyD2H() {

    // chequeo que esten alocadas las memorias en host y device

    if (d_allocated && h_allocated) {
        
        cudaMemcpy(data_h.get(), data_d.get(), shape.x*shape.y*sizeof(double),cudaMemcpyDeviceToHost);
        NNExc::thIfDevErr("No se pudo copiar los datos de device a host");
        
    }  
    else {
        // no hay memoria alocada, hago un throw para controlar errores
        NNExc::thIfDevErr("No se puede copiar los datos porque la memoria no se halla alocada : D2H");
    }
}

// equivalente a __get__ en la matriz de host (accedo al indice de la matriz)
double& Matrix::operator[](const int idx){
    return data_h.get()[idx];
}

const double& Matrix::operator[](const int idx) const{
    return data_h.get()[idx];
}