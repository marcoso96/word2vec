#include "costfun.cu"
#include "capas/matrix.hh"

using namespace std;

// siempre inicializo matrices en zeros
int main()
{
    float *loss = (float *)malloc(sizeof(float));
    size_t embed_size = 2;
    size_t outside_window = 3;

    W2VCost cost;
    // supongamos que acá magicamente me pasaron los vectores
    Matrix gradOutside(outside_window, embed_size);
    Matrix gradCenter(embed_size, 1);

    Matrix centerWordVec(embed_size, 1);
    Matrix outsideWordVecs(outside_window, embed_size);

    gradOutside.allocateMemory();
    gradCenter.allocateMemory();

    centerWordVec.allocateMemory();
    outsideWordVecs.allocateMemory();

    outsideWordVecs[0] = 1;
    outsideWordVecs[1] = 0;
    outsideWordVecs[2] = 0;
    outsideWordVecs[3] = 0;
    outsideWordVecs[4] = 1;
    outsideWordVecs[5] = 0;

    centerWordVec[0] = 1;
    centerWordVec[1] = 0;
    centerWordVec[2] = 1;

    outsideWordVecs.copyH2D();
    centerWordVec.copyH2D();

    cost.lossAndGrad(centerWordVec, outsideWordVecs, 1, loss, gradCenter, gradOutside);


    // free(loss);
    //acá magicamente los vectores se van a python
}