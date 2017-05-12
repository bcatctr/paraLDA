#include <cstring>
#include "mpi.h"
#include "Log.h"
#include "Master.h"
#include "Utils.h"

Master::Master(int _length) : length(_length) {
    global_table = new int[length];
    buf = new int[length];
    memset(global_table, 0, sizeof(int) * length);
    memset(buf, 0, sizeof(int) * length);
}

Master::~Master() {
    delete[] global_table;
    delete[] buf;
}

void Master::run() {
    MPI_Status status, recv_status;
    //MPI_Request req;
    int tmp;
    LOG("start run master\n");
    while (true) {
        LOG("waiting for message\n");
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        LOG("receive message\n");
        int tag = status.MPI_TAG;
        int source = status.MPI_SOURCE;
        LOG("tag: %d\n", tag);
        if (tag == COMM_COMPLETE_TAG) break;
        if (tag == COMM_FETCH_TAG) {
            LOG("fetch data from source: %d\n", source);
            MPI_Recv(&tmp, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Send(global_table, length, MPI_INT, source, 0, MPI_COMM_WORLD);
            LOG("finish fetch data\n");
        }
        else {
            LOG("start get local update\n");
            MPI_Recv(buf, length, MPI_INT, source, status.MPI_TAG, MPI_COMM_WORLD, &recv_status);
            // TODO: multi-threading update
            for (int i=0; i<length; i++) {
                global_table[i] += buf[i];
            }
            LOG("finish get local update\n");
        }
    }
    LOG("finish run master\n");
}
