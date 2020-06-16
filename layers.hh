#pragma once

#include <fstream>
#include <iostream>
#include <exception>

#include "matrix.hh"
// #include <cuda_runtime.h>

#define DIM 1024

//  GRACIAS http://luniak.io/cuda-neural-network-implementation-part-1/
// en este código se van a ver muchos puntos en los cuáles me maravillo (o muestro mi odio) por C++

// cada clase para cada capa requiere una propagación forward y backward
// uso polimorfismos para cagarme en que hace realmente cada capa

class NNLayer {

    // los miembros protegidos son accesibles desde la clase o quienes hereden sus características
    protected :

        std::string name;

    public: 

        // las funciones virtuales se establecen cuando uno las pone, por lo que no existen. Ja, son virtuales. Puedo hacer cosas sin que realmente la clase sepa acerca de como implementan las subclases los métodos

        virtual ~NNLayer() = 0;

        // defino las matrices virtuales que devuelven los métodos forward y backward de cada capa / lr es learning rate
        virtual Matrix& forward(Matrix& A) = 0;
        virtual Matrix& backprop(Matrix& dZ, float lr) = 0;
        
        std::string getName() { return this->name;}
};
// todavia no termino muy bien de entender esto, pero es muy util para que el llamado a la función no tenga overhead
// destructor de la capa
inline NNLayer::~NNLayer() { printf("NNLayer destructor\n"); }

// heredo las caracteristicas de una excepción 
// en este header declaré todo
class NNExc : std::exception {

private: 

    const char *exc_msg;

public: 
        
    NNExc(const char *exc_msg) :
            exc_msg(exc_msg)
    {}
    
    // que hago con un throw, lo tiro por aca
    // https://es.cppreference.com/w/cpp/error/exception/what devuelve un string explicando que onda
    virtual const char* what() const throw()
    {
        return exc_msg;
    }
    
    static void thIfDevErr(const char* exc_msg) 
    {   
        // agarro el ultimo error en el ERRNO de la GPU
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {

            std::cerr << error << " : "<< exc_msg;
            throw NNExc(exc_msg);
        }
    }
};
