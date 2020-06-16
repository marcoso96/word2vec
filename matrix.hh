#pragma once
#include <memory>


struct Shape {
	size_t x, y;

	Shape(size_t x = 1, size_t y = 1);
};

// defino una clase matrix en dónde realizo internamente todo el manejo de memoria h2d y d2h

class Matrix 
{
    private: 
        bool d_allocated;
        bool h_allocated;

        void allocateHostMemory();
        void allocateDevMemory();
    
    public: 
        // por como se declaró ya está cheta esta
        Shape shape;

        // van a encargarse de las referencias de memoria intelimágicamente
        std::shared_ptr<float> data_d;
        std::shared_ptr<float> data_h;

        // constructor
        Matrix(size_t x_dim = 1, size_t y_dim = 1);
        Matrix(Shape shape);

        // alocación de memoria en host y device
        void allocateMemory();
        void allocateMemoryIfNAll(Shape shape);

        void initWithZeros();
        
        void copyH2D();
        void copyD2H();

        // acceso al elemento, equivalente a get
        float& operator[](const int index);
        const float& operator[](const int index) const;

        void print_matrix();
};
