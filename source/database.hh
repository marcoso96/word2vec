#include "cnpy.h"
#include <vector>
#include "assert.h"
#include "matrix.hh"
#include "w2vembedding.hh"

#define SAVE_STEP 100

using namespace std;

class Database {
private:

    W2VEmbedding *dictionary;
    
    int* sents;
    
    // sentences contiene una matrix de [num_sent, max_len sent] con enteros
    // representando la base de datos sobre la cual entrenar los vectores

    string data_path;
    string metadata_path;

    int context;    // contexto para armar este diccionario
    int train_sents;
    int batch_size;

    // parámetros para oraciones
    int sentID;
    int wordID;
    int low_bound;
    int up_bound;

    void loadMetadata();
    void constructDictionary(double lr);

    int poissonRandom(double expectedValue);
    void getRandomContext();
    // carga las oraciones ejemplo en Sentences
    void loadSentences();
    int rand_lim(int limit);

public:

    int word_count;  // palabras en el diccionario
    int max_len;    // largo máximo de una oración
    int embed_size; // tamaño del vector
    int tot_sents;  // número de oraciones en base de datos

    Database(string data_path, string metadata_path, int train_sents, int context, int batch_size, double lr);   
    ~Database();
    void updateDictionary();
    void saveDictionary(string data_path);
};

