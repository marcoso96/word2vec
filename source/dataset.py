#%%
import numpy as np
import random
import subprocess
import shlex
import matplotlib.pyplot as plt
from itertools import product
import cupy as cp

class W2VDataset():
    
    def __init__(self, embed_size, path=None, delim = None):
        
        if not path:
            path = ""

        self.delim = delim
        self.path = path
        self.embed_size = embed_size
        self._sentences = []
        self._rejectProb = None
        self._allsentences = []
        self.dictionary = []
        self._numSentences = None
        
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
    
    def numSentences(self):
        
        if hasattr(self, "_numSentences") and self._numSentences:
        
            return self._numSentences
        
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences
    
    def sentences(self):
        
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        
        sentences = []
        with open("./" + self.path, "r") as f:
            
            first = True
            for line in f:
                # me salto la primera linea
                if first:
                    first = False
                    continue
                
                splitted = line.strip().replace(self.delim, ' ').split()[1:]
                sentences += [[w.lower() for w in splitted]]
        
        
        self._sentences = [s for s in sentences if len(s)>1]
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)
        
        
        return self._sentences
    
    def allSentences(self):
        
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        tokens = self.tokens()
        rejectProb = self.rejectProb()

        # filtro por aparición - si aparecen mucho, las hago mierda
        
        allsentences = [[w for w in s if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]] for s in sentences]
        
        
        self.max_len = max(self._sentlengths)
        allsentences = [[tokens[w] for w in s]+[-1 for i in range(self.max_len-len(s))] for s in allsentences if len(s) > 1 and len(s)<self.max_len]

        self._allsentences = allsentences
        self.tot_sents = len(allsentences)
        
        return self._allsentences
    
    def rejectProb(self):
       
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-1 * self._wordcount

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
        
        dictionary = cp.load(strpath)
        self.dictionary = cp.concatenate((dictionary[:self._wordcount,:], dictionary[self._wordcount:,:]), axis = 1)

    def getDict(self, strpath = None):
        """
        Devuelve el diccionario
        """
        self.loadDict(strpath)
        
        return self.dictionary
        
    def word2idx(self, str):
         
        """
        Devuelve el indice de la palabra en el diccionario
        """
        return self._tokens.get(str)

    def visualizeWords(self, words):
        
        visualIdx = [self._tokens[word] for word in words ]
        
        visualVecs = self.dictionary[visualIdx, :]

        temp = (visualVecs - np.mean(visualVecs, axis=0))
        covariance = 1.0 / len(visualIdx) * temp.T.dot(temp)
    
        U, _, _ = np.linalg.svd(covariance)
        coord = temp.dot(U[:, 0:2])    
        
        for i in range(len(words)):
            plt.text(coord[i,0], coord[i,1], words[i],
            bbox=dict(facecolor='green', alpha=0.1))

        plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
        plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

def lossPlot(str_loss):
            
    loss = np.loadtxt("out_loss.txt", delimiter = ",")

    plt.figure()
    plt.plot(loss[:,0], loss[:,1])
    plt.show()
    
    plt.savefig(str_loss)

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    
import gif

def wordCoords(words, database):
    
    visualIdx = [database._tokens[word] for word in words]

    visualVecs = database.dictionary[visualIdx, :]

    temp = (visualVecs - cp.mean(visualVecs, axis=0))
    covariance = 1.0 / len(visualIdx) * temp.T.dot(temp)
    U, _, _ = cp.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])   

    return coord

@gif.frame 
def plot_words(words, coord, frame_num):
    
    np_coord = cp.asnumpy(coord)
    plt.figure(figsize=(10, 10), dpi = 100)
    for i in range(len(words)):
            plt.text(np_coord[i,0], np_coord[i,1], words[i],
            bbox=dict(facecolor='green', alpha=0.1))
    plt.text( np.max(np_coord[:,0]), np.max(np_coord[:,1]),  str(frame_num) ,bbox=dict(facecolor='blue', alpha=0.5))
    plt.xlim((np.min(np_coord[:,0]), np.max(np_coord[:,0])))
    plt.ylim((np.min(np_coord[:,1]), np.max(np_coord[:,1])))
    plt.show()
    
#%% Cargado de base de datos

prueba = W2VDataset(embed_size = 50, path = 'news_database.txt', delim = ",")
prueba.allSentences()
prueba.toMat()
#%% Calculo tiempos en función de embed size, contexto y oraciones de entrenamiento

embed_size = [10]
train_sents = [100, 200, 500, 1000]
context = [5]
lr = 0.3
batch_size = 50
loss = []

for embed, train, ctx in product(embed_size, train_sents, context):
    
    prueba = W2VDataset(embed_size = embed,path = 'news_database.txt', delim = ",")
    prueba.allSentences()
    prueba.toMat()
    
    run = ["./dataprueba", str(train), str(ctx), str(lr), str(batch_size)]

    for path in execute(run):
        print(path, end="")
    
    loss.append(np.loadtxt("out_loss_Cublas.txt", delimiter = ","))
#%% Analizo tiempos de cálculo para distintos 

lista = np.array([ctx for _, _, ctx in product(embed_size, train_sents, context)])
lista = lista.reshape(len(lista), 1)
#%% Analizo tiempos de cálculo para distintos 

from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png
import pandas as pd


t_CUBLAS = np.loadtxt("Mediciones/TiemposCUDASIMPLE.txt", delimiter=",")
t_CUBLAS = np.append(t_CUBLAS, lista, axis = 1)

t_CUDA = np.loadtxt("Mediciones/TiemposCUBLAS.txt", delimiter=",")
t_CUDA = np.append(t_CUDA, lista, axis = 1)
#%% Analizo tiempos de cálculo para distintos 

df_t_CUBLAS = pd.DataFrame({'tiempos': t_CUBLAS[:,0], 'embed_size':t_CUBLAS[:,2], 'train_sent' : t_CUBLAS[:,1], 'context' : t_CUBLAS[:,3]})
df_t_CUBLAS = df_t_CUBLAS.astype({'tiempos': np.float64, 'embed_size':int, 'train_sent' : int, 'context' :int})

df_t_CUDA = pd.DataFrame({'tiempos': t_CUDA[:,0], 'embed_size':t_CUDA[:,2], 'train_sent' : t_CUDA[:,1], 'context' : t_CUDA[:,3]})
df_t_CUDA = df_t_CUDA.astype({'tiempos': np.float64, 'embed_size':int, 'train_sent' : int, 'context' :int})

#%% Analizo tiempos de cálculo para distintos 

df_train = df_t_CUBLAS[(df_t_CUBLAS["context"] == 2)]


plt.figure(figsize = (10, 10))
plt.title("")
df_train.set_index("train_sent", inplace = True)
df_train.groupby('embed_size')["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig("Mediciones/NumSent_Tiempo_CUBLAS_Embeds_C2.png")
plt.show()

df_train = df_t_CUBLAS[(df_t_CUBLAS["embed_size"] == 10)]
plt.figure(figsize = (10, 10))
plt.title("")
df_train.set_index("train_sent", inplace = True)
df_train.groupby('context')["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig("Mediciones/NumSent_Tiempo_CUBLAS_Context_E10.png")
plt.show()

#%%
df_train = df_t_CUDA[(df_t_CUDA["context"] == 2)]
plt.figure(figsize = (10, 10))
plt.title("")
df_train.set_index("train_sent", inplace = True)
df_train.groupby('embed_size')["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig("Mediciones/NumSent_Tiempo_CUDA_Embeds_C2.png")
plt.show()

df_train = df_t_CUDA[(df_t_CUDA["embed_size"] == 10)]
plt.figure(figsize = (10, 10))
plt.title("")
df_train.set_index("train_sent", inplace = True)
df_train.groupby('context')["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig("Mediciones/NumSent_Tiempo_CUDA_Context_E10.png")
plt.show()

#%% corro sobre headlines

prueba = W2VDataset(embed_size = 10, path = 'news_database.txt', delim = ',')
prueba.allSentences()
prueba.toMat()

#%% visualizo parámetros
len(prueba._sentences)
#%% run
train = 10000
ctx = 4
lr = 0.3
batch_size = 50

run = ["nvprof", "--csv", "output.csv","./dataprueba", str(train), str(ctx), str(lr), str(batch_size)]

for path in execute(run):
    print(path, end="")
#%%  Analizo la posición de los vectores mediante una reducción de la dimensionalidad

visualWords = ["death", "suspicious", "survivors", "plane", "police", "crash", "man", "kangaroos"]
frames = []
frame_num = 100

for i in range(frame_num):
    
    visVecs = prueba.getDict("Mediciones/News/news"+str(i)+".npy")
    coords = wordCoords(visualWords, prueba)
    
    frame = plot_words(visualWords, coords, i)
    frames.append(frame)
#%%
gif.save(frames, "Mediciones/Vectores1.gif", duration=100)    
#%%
visualWords = ["death", "survivors", "plane", "police", "crash", "man", "kangaroos", "road", "health", "health", "snow"]
visualIdx = [prueba._tokens[word] for word in visualWords]

X_1 = 0
visVecs = prueba.getDict("Mediciones/News/news"+str(X_1)+".npy")
coords = wordCoords(visualWords, prueba)
frame = plot_words(visualWords, coords, X_1)
visualVecs = prueba.dictionary[visualIdx, :]
simil = cp.dot(prueba.dictionary[prueba._tokens["plane"], :], prueba.dictionary[prueba._tokens["road"], :])/(cp.linalg.norm(prueba.dictionary[prueba._tokens["plane"], :])*cp.linalg.norm(prueba.dictionary[prueba._tokens["road"], :]))
print(simil)

X_2 = 40
visVecs = prueba.getDict("Mediciones/News/news"+str(X_2)+".npy")
coords = wordCoords(visualWords, prueba)
frame = plot_words(visualWords, coords, X_2)
visualVecs = prueba.dictionary[visualIdx, :]
simil = cp.dot(prueba.dictionary[prueba._tokens["plane"], :], prueba.dictionary[prueba._tokens["road"], :])/(cp.linalg.norm(prueba.dictionary[prueba._tokens["plane"], :])*cp.linalg.norm(prueba.dictionary[prueba._tokens["road"], :]))
print(simil)
#%% Busco similaridad

prueba._wordcount
#%% probadito

comp = ["nvcc", "-o", "dataprueba", "maindatabase.cu", "costfun.cu", "database.cu", "w2vembedding.cu", "matrix.cu", "--std=c++11",  "-L/usr/local/lib/", "-lcnpy", "-lz", "--std=c++11", "-lcurand"]
run = ["./dataprueba", "10000", "5"]

for path in execute(run):
    print(path, end="")

#%% LOSS
import pandas as pd

t_CUBLAS_1800 = np.loadtxt("Mediciones/TiemposNews1828_CUBLAS.txt", delimiter=",")
t_CUBLAS_7000 = np.loadtxt("Mediciones/TiemposNews7000_CUBLAS.txt", delimiter=",")
t_CUDA_1800 = np.loadtxt("Mediciones/TiemposNews1828_CUDA.txt", delimiter=",")
t_CUDA_7000 = np.loadtxt("Mediciones/TiemposNews7000_CUDA.txt", delimiter=",")

df_t_CUBLAS_1800 = pd.DataFrame({'CUBLAS': t_CUBLAS_1800[:4,0], 'train_sent' : t_CUBLAS_1800[:4,1]})
df_t_CUBLAS_1800 = df_t_CUBLAS_1800.astype({'CUBLAS': np.float64,  'train_sent' : int})
df_t_CUBLAS_7000 = pd.DataFrame({'CUBLAS': t_CUBLAS_7000[:,0], 'train_sent' : t_CUBLAS_7000[:,1]})
df_t_CUBLAS_7000 = df_t_CUBLAS_7000.astype({'CUBLAS': np.float64,  'train_sent' : int})

df_t_CUDA_1800 = pd.DataFrame({'CUDA kernel': t_CUDA_1800[:4,0], 'train_sent' : t_CUDA_1800[:4,1]})
df_t_CUDA_1800 = df_t_CUDA_1800.astype({'CUDA kernel': np.float64,  'train_sent' : int})
df_t_CUDA_7000 = pd.DataFrame({'CUDA kernel': t_CUDA_7000[:,0], 'train_sent' : t_CUDA_7000[:,1]})
df_t_CUDA_7000 = df_t_CUDA_7000.astype({'CUDA kernel': np.float64,  'train_sent' : int})

plt.figure(figsize = (10, 10))
plt.title("")
df_t_CUBLAS_1800.set_index("train_sent", inplace = True)
df_t_CUBLAS_7000.set_index("train_sent", inplace = True)
df_t_CUDA_1800.set_index("train_sent", inplace = True)
df_t_CUDA_7000.set_index("train_sent", inplace = True)

df_t_CUBLAS_1800["CUBLAS"].plot(loglog=True, style = "g.", markersize = 20, fontsize = 20)
df_t_CUBLAS_7000["CUBLAS"].plot(loglog=True, style = "r.", markersize = 20, fontsize = 20)
df_t_CUDA_1800["CUDA kernel"].plot(loglog=True, style = "g*", markersize = 20, fontsize = 20)
df_t_CUDA_7000["CUDA kernel"].plot(loglog=True, style = "r*", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.grid(True)
plt.legend()
plt.savefig("Mediciones/NumSent_Tiempo_Tots_News.png")
plt.show()


#%%
t_CUDA = np.loadtxt("Mediciones/TiemposNews_CUDA.txt", delimiter=",")

df_t_CUDA = pd.DataFrame({'tiempos': t_CUDA[:,0], 'train_sent' : t_CUDA[:,1]})
df_t_CUDA = df_t_CUDA.astype({'tiempos': np.float64,  'train_sent' : int})

plt.figure(figsize = (10, 10))
plt.title("")
df_t_CUDA.set_index("train_sent", inplace = True)
df_t_CUDA["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
df_t_CUBLAS["tiempos"].plot(loglog=True, style = ".", markersize = 20, fontsize = 20)
plt.xlabel("# de oraciones", fontsize = 20)
plt.ylabel("Tiempo [s]", fontsize = 20)
plt.grid(True)
plt.savefig("Mediciones/NumSent_Tiempo_CUDA_News.png")
plt.show()
#%%
df_t_CUDA_1800["CUDA kernel"]