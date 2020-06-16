#include "database.hh"
#include "matrix.hh"
#include "w2vembedding.hh"
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

int main()
{   
    int batch_size = 1;
    int embed_size = 5;
    int vocab_size = 3;
    int window_size = 3;

    Shape dict_shape(vocab_size, embed_size);
    W2VEmbedding dict(dict_shape);

    srand(time(NULL));

    for(i = 0; i<batch_size; i++)
    {
        window = rand()%window_size;
    
    }
    return 0;
}


    // Database data;

    // data.loadSentences("dataset.npy", "data_words.npy");

    // print_matrix(data.sents, data.tot_sents, data.max_len);

    // return 0;