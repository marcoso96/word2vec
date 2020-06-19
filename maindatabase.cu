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

int main()
{   
    int train_sents = 3;
    int context = 2;

    srand(time(NULL));
    Database data("dataset.npy", "data_words.npy", context, train_sents);
    
    for(int i=0; i<train_sents; i++)
    {
        data.getRandomContext();
    }
        
    return 0;
}


    // Database data;

    // data.loadSentences("dataset.npy", "data_words.npy");

    // print_matrix(data.sents, data.tot_sents, data.max_len);

    // return 0;