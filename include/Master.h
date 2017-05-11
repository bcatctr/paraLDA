#ifndef PARALDA_MASTER_H
#define PARALDA_MASTER_H

class Master {
    int length;
    int* global_table;
    int* buf;

public:
    Master(int _length);
    ~Master();
    void run();
};

#endif //PARALDA_MASTER_H
