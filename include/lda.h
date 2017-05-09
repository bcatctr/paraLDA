//
// Created by CHEN HU on 4/23/17.
//

#ifndef PARALDA_LDA_H
#define PARALDA_LDA_H


#include <string>
#include <vector>
#include <random>
#include "dataLoader.h"

class lda {
    int num_topics;
    int num_docs;
    int vocab_size;
    double alpha;
    double beta;
    int num_iterations;
    dataLoader* data_loader;
    int memory_size;
    int* local_table_memory;
    int* global_table_memory;
    int* local_topic_table;
    int* global_topic_table;
    int** local_topic_word_table;
    int** global_topic_word_table;
    int** doc_topic_table;
    std::vector<std::vector<int>> W;
    std::vector<std::vector<int>> T;
    std::string output;

    int rank;
    int comm_size;


    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

    void reduce_tables();


public:
    lda(std::string dataFile, std::string output, int num_topics,
        double alpha, double beta, int num_iterations, int rank, int comm_size);
    ~lda();
    void initialize();
    void runGibbs();
    int resample(std::vector<double> dis);
    double getLogLikelihood();
    double logDirichlet(double* X, int N);
    double logDirichlet(double x, int N);
    void printTopicWord();
    void printDocTopic();

};


#endif //LDA_LDA_H