#ifndef PARALDA_MASTER_H
#define PARALDA_MASTER_H

class Master {
    int length;
    int* global_table;
    int* buf;
    int rank;
    int worker_size;
    double* llh;
    double glh;
    int master_count;
    double threshold;
    double elapsed_time;
    double last;

public:
    Master(int _length, int _rank, int _worker_size, int _master_count, double _threshold);
    ~Master();
    void run();
    void watchdog();
};

#endif //PARALDA_MASTER_H
