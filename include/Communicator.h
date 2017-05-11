#ifndef PARALDA_COMMUNICATOR_H
#define PARALDA_COMMUNICATOR_H

#include <mpi.h>

class Communicator {
    int master_cnt;
    MPI_Request* req;

public:
    Communicator(int _master_cnt);
    ~Communicator();

    void ISend(int* buf, int length);
    void IRecv(int* buf, int length);
    void Recv(int* buf, int length);
    void Complete();
    bool Test();
};

#endif //PARALDA_COMMUNICATOR_H
