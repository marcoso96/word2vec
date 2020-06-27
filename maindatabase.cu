#include "database.hh"
#include <iostream>
#include <time.h>
#include <stdlib.h>

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

    assert(argc == 3);
    train_sents = atoi(argv[1]);
    context = atoi(argv[2]);

    srand(time(NULL));
    Database data("dataset.npy", "data_words.npy", train_sents, context, 0.3);
    
    data.updateDictionary();
    data.saveDictionary("palabritas.npy");

    return 0;
}
