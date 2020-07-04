#include "database.hh"
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "cpu_timer.h"

using namespace std;

void print(int * input, int max)
{   
    for (int i=0; i<max; i++)
    {
        cout << input[i] << ' ';
    }
    cout << endl;
}

int main(int argc, char *argv[])
{   
    int train_sents;
    int context;
    int batch_size;
    double lr;
    ofstream tiempo;
    cpu_timer reloj_cpu;

    tiempo.open("Mediciones/TiemposNews.txt", ofstream::out | ofstream::app);

    // Uso : ./dataprueba train_sents context lr batch_size
    assert(argc == 5);
    train_sents = atoi(argv[1]);    
    context = atoi(argv[2]);
    lr = atof(argv[3]);
    batch_size = atoi(argv[4]);

    srand(time(NULL));
    Database data("dataset.npy", "data_words.npy", train_sents, batch_size, context, lr);

    reloj_cpu.tic();
    data.updateDictionary();
    tiempo << reloj_cpu.tac() << "," << train_sents << "," << data.embed_size << endl;
    
    data.saveDictionary("palabritasNews.npy");

    tiempo.close();
    return 0;
}
