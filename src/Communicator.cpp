#include <algorithm>
#include "Log.h"
#include "Communicator.h"
#include "Utils.h"

Communicator::Communicator(int _master_cnt)
        : master_cnt(_master_cnt) {
    reqs = new MPI_Request[master_cnt];
    statuses = new MPI_Status[master_cnt];
}

Communicator::~Communicator() {
    delete[] reqs;
    delete[] statuses;
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
        MPI_Irecv(buf + i * block_size, std::min(length, block_size), MPI_INT, i, 0, MPI_COMM_WORLD, &reqs[i]);
        length -= block_size;
    }
}

void Communicator::Recv(int *buf, int length) {
    int block_size = length / master_cnt;
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
        MPI_Test(&reqs[i], &flag, &status);
        if (!flag) {
            return false;
        }
    }
    return true;
}

void Communicator::Wait() {
    MPI_Waitall(master_cnt, reqs, statuses);
}

void Communicator::Complete() {
    for (int i=0; i<master_cnt; i++) {
        MPI_Send(&i, 1, MPI_INT, i, COMM_COMPLETE_TAG, MPI_COMM_WORLD);
    }
}
