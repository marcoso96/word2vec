#include "cnpy.h"
#include <vector>
#include "assert.h"
#include "matrix.hh"
#include "w2vembedding.hh"

using namespace std;

class Database {
private:

    int* sents;

    int word_count;  // palabras en el diccionario
    int max_len;    // largo máximo de una oración
    int embed_size; // tamaño del vector
    int tot_sents;  // número de oraciones en base de datos
    
    // sentences contiene una matrix de [num_sent, max_len sent] con enteros
    // representando la base de datos sobre la cual entrenar los vectores

    string data_path;
    string metadata_path;

    int context;    // contexto para este diccionario

    // parámetros para oraciones
    int sentID;
    int wordID;
    int low_bound;
    int upper_bound;

    W2VEmbedding dictionary;

    void loadMetadata();
    void constructDictionary();

    // carga las oraciones ejemplo en Sentences
    void loadSentences();
    void getRandomContext();

public:

    Database(string data_path, string metadata_path, int context);   
};