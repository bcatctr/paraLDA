#include <mpi.h>
#include "Master.h"
#include "Utils.h"

Master::Master(int _length) : length(_length) {
    global_table = new int[length];
    buf = new int[length];
}

Master::~Master() {
    delete[] global_table;
    delete[] buf;
}

void Master::run() {
    MPI_Status status, recv_status;
    MPI_Request req;
    int tmp;
    while (true) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;
        if (tag == COMM_COMPLETE_TAG) break;
        if (tag == COMM_FETCH_TAG) {
            MPI_Recv(&tmp, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &recv_status);
            MPI_Isend(global_table, length, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &req);
        }
        else {
            MPI_Recv(buf, length, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &recv_status);
            // TODO: multi-threading update
            for (int i=0; i<length; i++) {
                global_table[i] += buf[i];
            }
        }
    }
}
