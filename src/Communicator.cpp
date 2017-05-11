#include <algorithm>
#include "Communicator.h"
#include "Utils.h"

Communicator::Communicator(int _master_cnt)
        : master_cnt(_master_cnt) {
    req = new MPI_Request[master_cnt];
}

Communicator::~Communicator() {
    delete[] req;
}

void Communicator::ISend(int *buf, int length) {
    int block_size = length / master_cnt;
    for (int i=0; i<master_cnt; i++) {
        MPI_Request req;
        MPI_Isend(buf + i * block_size, std::min(length, block_size), MPI_INT, i, 0, MPI_COMM_WORLD, &req);
        length -= block_size;
    }
}

void Communicator::IRecv(int *buf, int length) {
    int block_size = length / master_cnt;
    for (int i=0; i<master_cnt; i++) {
        MPI_Request req;
        MPI_Isend(&block_size, 1, MPI_INT, i, COMM_FETCH_TAG, MPI_COMM_WORLD, &req);
    }

    for (int i=0; i<master_cnt; i++) {
        MPI_Irecv(buf + i * block_size, std::min(length, block_size), MPI_INT, i, 0, MPI_COMM_WORLD, &req[i]);
        length -= block_size;
    }
}

void Communicator::Recv(int *buf, int length) {
    int block_size = length / master_cnt;
    for (int i=0; i<master_cnt; i++) {
        MPI_Request req;
        MPI_Isend(&block_size, 1, MPI_INT, i, COMM_FETCH_TAG, MPI_COMM_WORLD, &req);
    }

    for (int i=0; i<master_cnt; i++) {
        MPI_Status status;
        MPI_Recv(buf + i * block_size, std::min(length, block_size), MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        length -= block_size;
    }
}

bool Communicator::Test() {
    int flag;
    for (int i=0; i<master_cnt; i++) {
        MPI_Status status;
        MPI_Test(&req[i], &flag, &status);
        if (!flag) return false;
    }
    return true;
}

void Communicator::Complete() {
    for (int i=0; i<master_cnt; i++) {
        MPI_Send(&i, 1, MPI_INT, i, COMM_COMPLETE_TAG, MPI_COMM_WORLD);
    }
}
