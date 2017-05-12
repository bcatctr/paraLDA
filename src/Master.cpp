#include <cstring>
#include "mpi.h"
#include "Log.h"
#include "Master.h"
#include "Utils.h"
#include <omp.h>
#include <thread>
#include <cmath>
#include <ctime>

Master::Master(int _length, int _rank, int _worker_size, int _master_count, double _threshold)
        : length(_length), rank(_rank), worker_size(_worker_size), master_count(_master_count), threshold(_threshold) {
    LOG("length: %d\n", length);
    global_table = new int[length];
    buf = new int[length];
    memset(global_table, 0, sizeof(int) * length);
    memset(buf, 0, sizeof(int) * length);
    llh = new double[worker_size];
    memset(llh, 0, sizeof(double) * worker_size);
    elapsed_time = 0;
    last = 0;
}

Master::~Master() {
    delete[] global_table;
    delete[] buf;
    delete[] llh;
}

void Master::watchdog() {
    double old = 1000;
    int iter = 0;
    bool flag = false;
    while (true) {
        if (std::fabs(llh[0] - last) > 1e-6) {
            last = llh[0];
            double lh = glh;
            for (int i = 0; i < worker_size; i++) {
                lh += llh[i];
            }
            if (old < 0) {
                double diff = std::fabs((old - lh) / lh);

                LOG("Iteration:%d\tloglikelihood:%.8f\trel_change:%.4f\ttime:%.2f\ttotal_time:%.2f\n", iter, lh, diff,
                    elapsed_time / (iter + 1), elapsed_time);
                iter++;

                if (diff < threshold) {
                    if (flag) return;
                    flag = true;
                }
            }
            old = lh;
        }
        std::this_thread::sleep_for (std::chrono::milliseconds(100));
    }
}

void Master::run() {
    if (rank == 0) {
        std::thread t([this]() {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            watchdog();
        });
        t.detach();
    }
    MPI_Status status, recv_status;
    //MPI_Request req;
    LOG("start run master\n");
    int finish_cnt = 0;
    int tmp;
    double msg[3];
    while (true) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;
        int source = status.MPI_SOURCE;
        switch (tag) {
            case COMM_COMPLETE_TAG:
                MPI_Recv(&tmp, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &recv_status);
                if (++finish_cnt >= worker_size) {
                    LOG("finish run master\n");
                    return;
                }
                break;
            case COMM_LLH:
                MPI_Recv(msg, 3, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &recv_status);
                llh[source - master_count]  = msg[0];
                glh = msg[1];
                elapsed_time = msg[2];
                break;
            default:
                MPI_Recv(buf, length, MPI_INT, source, 0, MPI_COMM_WORLD, &recv_status);
                // TODO: multi-threading update
#pragma omp parallel for schedule(static, 64)
                for (int i = 0; i < length; i++) {
                    global_table[i] += buf[i];
                }
                MPI_Send(global_table, length, MPI_INT, source, 0, MPI_COMM_WORLD);
        }
    }
}
