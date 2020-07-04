APPS=dataprueba

all: ${APPS}

dataprueba: source/maindatabase.cu
	nvcc source/maindatabase.cu source/costfun.cu source/database.cu source/w2vembedding.cu source/matrix.cu --std=c++11  -Lsource/ -L/usr/local/lib/ -lcnpy -lz --std=c++11 -lcurand -lcublas -o dataprueba

clean:
	rm -f ${APPS}

submit:	all
	qsub jobGPU; watch qstat
