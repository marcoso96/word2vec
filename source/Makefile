APPS=dataprueba

all: ${APPS}

dataprueba: maindatabase.cu
	nvcc maindatabase.cu costfun.cu database.cu w2vembedding.cu matrix.cu --std=c++11  -L/usr/local/lib/ -lcnpy -lz --std=c++11 -lcurand -lcublas -o dataprueba

clean:
	rm -f ${APPS}

submit:	all
	qsub jobGPU; watch qstat
