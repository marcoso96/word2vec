#include "database.hh"

Database::Database(string data_path, string metadata_path, int context)
{
    this->data_path = data_path;
    this->metadata_path = metadata_path;

    // cargo los parámetros que necesito 
    loadMetadata();
    // cargo la base de datos de oraciones
    loadSentences();

    constructDictionary();
}

void Database::loadMetadata()
{   
    cnpy::NpyArray arr = cnpy::npy_load(metadata_path);
    
    int * params = arr.data<int>();

    this -> word_count  = params[0]; 
    this -> max_len    = params[1]; 
    this -> embed_size = params[2]; 
    this -> tot_sents  = params[3];
}

void Database::loadSentences()
{   

    cnpy::NpyArray arr = cnpy::npy_load(this->data_path);

    this -> sents = arr.data<int>();

}

void Database::constructDictionary()
{
    Shape dict_shape(word_count, embed_size);

    dictionary(dict_shape, context);
}


// agarro un contexto random de la base de datos
void Database::getRandomContext()
{
    sentID = rand()%tot_sents;
    wordID = rand()%max_len;    // center word
    
    low_bound = wordID - context;
    upper_bound = wordID + context;

    if (sents[sentID*max_len+wordID] == -1)
    {
        getRandomContext(context);
    }    
    //     // caso raro
    // if ((wordID - upper_bound)<0)
    // {
    //     getRandomContext(context);
    // }

    // a partir de acá, pongo cotas y devuelvo center
    if( (wordID - context) < 0)
    {
        low_bound = 0;
    }
    
    // busco la cota superior porque las oraciones estan paddeadas
    while (sents[sentID*max_len+upper_bound] == -1)
    {
        upper_bound--;
    }
}