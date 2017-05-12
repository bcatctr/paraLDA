#include <cstring>
#include "mpi.h"
#include "Log.h"
#include "Master.h"
#include "Utils.h"

Master::Master(int _length) : length(_length) {
    LOG("length: %d\n", length);
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
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;
        int source = status.MPI_SOURCE;
        if (tag == COMM_COMPLETE_TAG) break;
        if (tag == COMM_FETCH_TAG) {
            MPI_Recv(&tmp, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Send(global_table, length, MPI_INT, source, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Recv(buf, length, MPI_INT, source, 0, MPI_COMM_WORLD, &recv_status);
            // TODO: multi-threading update
            for (int i=0; i<length; i++) {
                global_table[i] += buf[i];
            }
        }
    }
    LOG("finish run master\n");
}
