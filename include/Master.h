#ifndef PARALDA_MASTER_H
#define PARALDA_MASTER_H

class Master {
    int length;
    int* global_table;
    int* buf;
    int comm_size;

public:
    Master(int _length, int comm_size);
    ~Master();
    void run();
};

#endif //PARALDA_MASTER_H
