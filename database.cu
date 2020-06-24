#include "database.hh"

using namespace std;

Database::Database(string data_path, string metadata_path, int context, int train_sents, int lr)   
{
    this->data_path = data_path;
    this->metadata_path = metadata_path;
    this->context = context;
    this->train_sents = train_sents;
    this -> lr = lr;

    // cargo los parámetros que necesito 
    loadMetadata();
    // cargo la base de datos de oraciones
    loadSentences();
    // con las oraciones, construyo el diccionario
    constructDictionary();
}

Database::~Database(){

    free(this->sents);
}

int Database::rand_lim(int limit) {
/* return a random number between 0 and limit inclusive.
 */

    int divisor = RAND_MAX/(limit+1);
    int retval;

    do { 
        retval = rand() / divisor;
    } while (retval > limit);

    return retval;
}

void Database::loadMetadata()
{   
    cnpy::NpyArray arr = cnpy::npy_load(metadata_path);
    
    int * params = arr.data<int>();

    this -> word_count  = params[0]; 
    this -> max_len     = params[1]; 
    this -> embed_size  = params[2]; 
    this -> tot_sents   = params[3];

    this -> sents = (int *)malloc(tot_sents*max_len*sizeof(int));
}

void Database::loadSentences()
{   
    cnpy::NpyArray arr = cnpy::npy_load(this->data_path);

    // luego habría que ver de stremear a device esto
    memcpy(this -> sents, arr.data<int>(), tot_sents*max_len*sizeof(int));
}

void Database::constructDictionary()
{
    Shape dict_shape(word_count, embed_size);

    this -> dictionary = new W2VEmbedding(dict_shape, context, tot_sents, train_sents, lr);
}

// agarro un contexto random de la base de datos
void Database::getRandomContext()
{   
    sentID = rand_lim(tot_sents-1)*max_len;
    wordID = rand_lim(max_len-1);    // center word
    
    low_bound = wordID - context;
    up_bound = wordID + context;
    // condiciones de borde / bad word -> saco una nueva palabra
    if ((sentID+low_bound)<0  || sents[sentID+wordID] == -1){
        
        getRandomContext();
        return;
    }

    // low_bound se sube a la oracion anterior
    while (low_bound < 0)
    {
        low_bound++;
    }
    
    // up_bound se mete en palabras invalidas
    while (sents[sentID+up_bound] == -1 || up_bound>max_len)
    {
        up_bound--;
    }

    if ((up_bound-low_bound)>0)
    {
        dictionary->updateDictionary(sents, sentID, wordID, low_bound, up_bound);
        // cout << "SentID : " << sentID/max_len << endl;
        // cout << "WordID : " << wordID << endl;
        // cout << "TotSents : " << tot_sents << endl;
        // cout << "Max Len : " << max_len << endl;
        // cout << "Up Bound : " << up_bound << endl;
        // cout << "Low Bound : " << low_bound << endl;
        
        // cout<< sents[sentID+wordID] << '\t';

        // cout<< endl;
        return;
    }
    else 
    {
        getRandomContext();
        return;
    }
}

