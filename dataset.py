#%%
import numpy as np
import random
import subprocess
import matplotlib.pyplot as plt

class W2VDataset():
    
    def __init__(self, embed_size, path=None):
        
        if not path:
            path = ""

        self.path = path
        self.embed_size = embed_size
        self._sentences = []
        self._rejectProb = []
        self._allsentences = []
        self.dictionary = []
        
    def tokens(self):
        
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0 
        
        for sentence in self.sentences():
            
            for w in sentence:
                
                if not w in tokens:
                    wordcount += 1
                    # me armo el diccionario
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1
        
        # token desconocido
        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1
        
        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        
        return self._tokens
    
    def sentences(self):
        
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        
        sentences = []
        with open(self.path + "dataset.txt", "r") as f:
            
            first = True
            for line in f:
                # me salto la primera linea
                if first:
                    first = False
                    continue
                
                splitted = line.strip().split()[1:]
                sentences += [[w.lower() for w in splitted]]
                
        self._sentences = [s for s in sentences if len(s)>1]
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)
        
        return self._sentences
    
    def allSentences(self):
        
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        
        self.max_len = max(self._sentlengths)
        # ver despues la forma de filtrar por probabilidad
        allsentences = [[tokens[w] for w in s]+[-1 for i in range(self.max_len-len(s))] for s in sentences]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences
        self.tot_sents = len(allsentences)
        
        return self._allsentences
        
    
    def rejectProb(self):
       
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 0 * self._wordcount

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb
    
    
    def toMat(self):
        
        np.save('dataset.npy', np.array(self._allsentences, dtype = np.int32).T)
        np.save('data_words.npy', np.array([self._wordcount, self.max_len, self.embed_size, self.tot_sents], dtype = np.int32).T)
        
    def loadDict(self, strpath = "palabritas.npy"):
        
        dictionary = np.load(strpath)
        self.dictionary = np.concatenate((dictionary[:self._wordcount,:], dictionary[self._wordcount:,:]), axis = 1)

    
    def getDict(self):
        """
        Devuelve el diccionario
        """
        return self.dictionary
        
    def word2idx(self, str):
         
        """
        Devuelve el indice de la palabra en el diccionario
        """
        return self._tokens.get(str)

    def visualizeWords(self, words):
        
        visualIdx = [self._tokens[word] for word in words ]
        
        visualVecs = self.dictionary[visualIdx, :]
        print(visualVecs)
        temp = (visualVecs - np.mean(visualVecs, axis=0))
        covariance = 1.0 / len(visualIdx) * temp.T.dot(temp)
                
        return covariance
    
            
#%% probadito

prueba = W2VDataset(embed_size = 10)
prueba.allSentences()
prueba.toMat()

comp = False

if comp == True:
    
    subprocess.run(["nvcc", "-o", "dataprueba", "maindatabase.cu", "costfun.cu", "database.cu", "w2vembedding.cu", "matrix.cu", "--std=c++11",  "-L/usr/local/lib/", "-lcnpy", "-lz", "--std=c++11", "-lcurand"])

subprocess.run(["./dataprueba", "20", "2"])
prueba.loadDict()
print(prueba.getDict())
# resolver tema de outside words

#%% Visualizo resultados

visualWords = ["buenardo", "es", "la"]
#%% Colormap

plt.figure(figsize = (20, 20))
plt.imshow(prueba.visualizeWords(visualWords))
plt.yticks(list(prueba._tokens.values()), list(prueba._tokens.keys()))

plt.show()
#%%
prueba._tokens.values()
#%%

