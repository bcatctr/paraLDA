//
// Created by CHEN HU on 4/23/17.
//

#include <iostream>
#include <math.h>
#include <fstream>
#include <Log.h>
#include "lda.h"
#include "CycleTimer.h"
#include "mpi.h"


lda::lda(std::string dataDir, std::string output, int num_topics,
         double alpha, double beta, int num_iterations, int rank, int comm_size)
        : gen(std::random_device()()), dis(0, 1) {
    this->rank = rank;
    this->comm_size = comm_size;
    this->num_topics = num_topics;
    this->alpha = alpha;
    this->beta = beta;
    this->num_iterations = num_iterations;
    this->data_loader = new dataLoader(dataDir, rank, comm_size);
    this->output = output;

    num_docs = data_loader->docsCount();
    vocab_size = data_loader->vocabSize();

    memory_size = num_topics * vocab_size + num_topics;

    local_table_memory = new int[memory_size];
    global_table_memory = new int[memory_size];

    local_topic_table = local_table_memory + num_topics * vocab_size;
    global_topic_table = global_table_memory + num_topics * vocab_size;

    local_topic_word_table = new int*[num_topics];
    global_topic_word_table = new int*[num_topics];
    int j = 0;
    for(int i = 0; i < num_topics; i++) {
        local_topic_word_table[i] = local_table_memory + j;
        global_topic_word_table[i] = global_table_memory + j;
        j += vocab_size;
    }


    doc_topic_table = new int*[num_docs];
    for(int i = 0; i < num_docs; i++)
        doc_topic_table[i] = new int[num_topics]();

    W = data_loader->loadCorpus();

    T.resize(W.size());
    for(int i = 0; i < (int)W.size(); i++){
        std::vector<int> temp(W[i].size(), -1);
        T[i] = temp;
    }
}

lda::~lda() {
    delete data_loader;

    delete[] global_table_memory;
    delete[] local_table_memory;

    delete[] global_topic_word_table;
    delete[] local_topic_word_table;

    for(int i = 0; i < num_docs; i++)
        delete[] doc_topic_table[i];
    delete[] doc_topic_table;

    T.clear();
}

void lda::initialize() {
    std::random_device rd;
    std::mt19937 int_gen(rd());
    std::uniform_int_distribution<> int_dis(0, num_topics - 1);

    for(int d = 0; d < (int) T.size(); d++){
        for(int j = 0; j < (int) T[d].size(); j++){
            int word = W[d][j];
            int topic = int_dis(int_gen);
            T[d][j] = topic;
            doc_topic_table[d][topic] ++;
            local_topic_word_table[topic][word] ++;
            local_topic_table[topic] ++;
        }
    }
}


void lda::reduce_tables() {
    int block_size = 1 << 23;
    if (memory_size <= block_size) {
        MPI_Allreduce(local_table_memory, global_table_memory, memory_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        int j = 0;
        for (; j+block_size<=memory_size; j+=block_size) {
            MPI_Allreduce(local_table_memory + j, global_table_memory + j,
                          block_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
        if (j < memory_size) {
            MPI_Allreduce(local_table_memory + j, global_table_memory + j,
                          memory_size - j, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }
}


void lda::runGibbs() {

    initialize();
    std::vector<double> dis((size_t)num_topics, 0);

    CycleTimer timer;
    for(int iter = 0; iter < num_iterations; iter++){
        reduce_tables();
        for(int d = 0; d < (int) W.size(); d++){
            for(int j = 0; j < (int) W[d].size(); j++){
                int word = W[d][j];
                int topic = T[d][j];
                // ignore current position
                doc_topic_table[d][topic] --;
                local_topic_word_table[topic][word] --;
                local_topic_table[topic] --;

                global_topic_word_table[topic][word] --;
                global_topic_table[topic]--;

                // recalculate topic distribution
                for(int k = 0; k < num_topics; k++) {
                    dis[k] = (global_topic_word_table[k][word] + beta) / (global_topic_table[k] + beta * vocab_size) * (doc_topic_table[d][k] + alpha);
                }

                topic = resample(dis);
                T[d][j] = topic;
                doc_topic_table[d][topic] ++;
                local_topic_word_table[topic][word] ++;
                local_topic_table[topic] ++;

                global_topic_word_table[topic][word] ++;
                global_topic_table[topic]++;
            }
        }
        double llh = getLogLikelihood();
        double global_llh = 0;
        MPI_Allreduce(&llh, &global_llh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        LOG("Iteration: %d, loglikelihood: %.8f, time: %.2fs\n", iter, global_llh, timer.get_time_elapsed());
    }
}


int lda::resample(std::vector<double> multi_dis) {

    // normalize
    double sum = 0;
    for(int i = 0; i < num_topics; i++){
        sum += multi_dis[i];
    }


    double prob = dis(gen)*sum;

    double accum = 0;
    for(int i = 0; i < num_topics; i++){
        accum += multi_dis[i];
        if(prob < accum)
            return i;
    }
    return num_topics - 1;

}

double lda::logDirichlet(double *X, int N) {
    double sumLogGamma = 0.0;
    double logSumGamma = 0.0;
    for(int i = 0; i < N ; i++){
        sumLogGamma += std::lgamma(X[i]);
        logSumGamma += X[i];
    }

    return sumLogGamma - std::lgamma(logSumGamma);
}

double lda::logDirichlet(double x, int N) {
    return N * std::lgamma(x) - std::lgamma(N * x);
}

double lda::getLogLikelihood() {
    double lik = 0.0;

    double* temp = new double[vocab_size];
    for(int k = 0; k < num_topics; k++){
        int* word_vector = global_topic_word_table[k];
        for(int w = 0; w < vocab_size; w++){
            temp[w] = word_vector[w] + beta;
        }
        lik += logDirichlet(temp, vocab_size);
        lik -= logDirichlet(beta, vocab_size);
    }
    delete[] temp;

    temp = new double[num_topics];
    for(int d = 0; d < num_docs; d++){
        int* topic_vector = doc_topic_table[d];
        for(int k = 0; k < num_topics; k++){
            temp[k] = topic_vector[k] + alpha;
        }
        lik += logDirichlet(temp, num_topics);
        lik -= logDirichlet(alpha, num_topics);
    }
    delete[] temp;

    return lik;
}

void lda::printTopicWord() {
    std::string fileName = "output/" + output + ".tw";
    std::ofstream out_file(fileName);
    for(int k = 0; k < num_topics; k++){
        for(int w = 0; w < vocab_size - 1; w++){
            out_file << global_topic_word_table[k][w] << ",";
        }
        out_file << global_topic_word_table[k][vocab_size - 1] << "\n";
    }
}

void lda::printDocTopic() {
    std::string fileName = "output/" + output + ".dt";
    std::ofstream out_file(fileName);
    for(int d = 0; d < num_docs; d++){
        for(int k = 0; k < num_topics - 1; k++){
            out_file << doc_topic_table[d][k] << ",";
        }
        out_file << doc_topic_table[d][num_topics - 1] << "\n";
    }

}
