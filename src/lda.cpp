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

    local_word_topic_table = new int*[vocab_size];
    global_word_topic_table = new int*[vocab_size];
    int j = 0;
    for(int i = 0; i < vocab_size; i++) {
        local_word_topic_table[i] = local_table_memory + j;
        global_word_topic_table[i] = global_table_memory + j;
        j += num_topics;
    }
    memset(local_table_memory, 0, sizeof(int) * memory_size);
    memset(global_table_memory, 0, sizeof(int) * memory_size);

    doc_topic_table.resize(num_docs);
    for(int i = 0; i < num_docs; i++){
        std::unordered_map<int, int> temp;
        doc_topic_table[i] = temp;
    }


    W = data_loader->loadCorpus();

    T.resize(W.size());
    for(int i = 0; i < (int)W.size(); i++){
        std::vector<int> temp(W[i].size(), -1);
        T[i] = temp;
    }

    vocab_temp = new double[vocab_size];
    topic_temp = new double[num_topics];

}

lda::~lda() {
    delete data_loader;

    delete[] global_table_memory;
    delete[] local_table_memory;

    delete[] global_word_topic_table;
    delete[] local_word_topic_table;

    doc_topic_table.clear();

    delete[] vocab_temp;
    delete[] topic_temp;

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
            local_word_topic_table[word][topic] ++;
            local_topic_table[topic] ++;
        }
    }
    reduce_tables();
}


void lda::reduce_tables() {
    memset(global_table_memory, 0, sizeof(int) * memory_size);
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

    std::vector<double> c((size_t)num_topics, 0);
    std::vector<double> f((size_t)num_topics, 0);
    std::vector<double> g((size_t)num_topics, 0);
    std::vector<double> e((size_t)num_topics, 0);
    double F,G,E,Q;
    double denominator;

    CycleTimer timer;
    // pre-calculate G and g because they are irrelevant with document
    G = 0;
    for(int k = 0; k < num_topics; k++){
        g[k] = beta * alpha / (global_topic_table[k] + beta * vocab_size);
        G += g[k];
    }

    for(int iter = 0; iter < num_iterations; iter++){

        for(int d = 0; d < (int) W.size(); d++){
            // calculate coefficient c which can be updated topic by topic in iterations by word

            // calculate "doc-topic" bucket F and coefficient c
            F = 0;
            std::fill(f.begin(),f.end(),0);
            std::unordered_map<int, int>& curr_doc = doc_topic_table[d];

            for(int k = 0; k < num_topics; k++){
                c[k] = (curr_doc[k] + alpha) / (global_topic_table[k] + beta * vocab_size);
            }
            for(auto const &ent : curr_doc){
                f[ent.first] = (beta * ent.second) / (global_topic_table[ent.first] + beta * vocab_size);
                F += f[ent.first];
            }

            /*
            for(int k = 0; k < num_topics ; k++){
                denominator = global_topic_table[k] + beta * vocab_size;
                c[k] = (curr_doc[k] + alpha) / denominator;
                if(curr_doc[k] == 0) continue;

                f[k] = (beta * curr_doc[k]) / denominator;
                F += f[k];
            }*/

            // sample a new topic for each word of the document
            for(int j = 0; j < (int) W[d].size(); j++){
                int word = W[d][j];
                int topic = T[d][j];
                // ignore current position, update all related value
                doc_topic_table[d][topic] --;
                local_word_topic_table[word][topic] --;
                local_topic_table[topic] --;

                global_word_topic_table[word][topic] --;
                global_topic_table[topic]--;

                denominator = global_topic_table[topic] + beta * vocab_size;
                F -= f[topic];
                f[topic] = (beta * curr_doc[topic]) / denominator;
                F += f[topic];
                c[topic] = (curr_doc[topic] + alpha) / denominator;
                G -= g[topic];
                g[topic] = (beta * alpha) / denominator;
                G += g[topic];

                // calculate ""topic-word" bucket E
                E = 0;
                std::fill(e.begin(),e.end(),0);
                int* curr_word = global_word_topic_table[word];
                for(int k = 0; k < num_topics; k++) {
                    if(curr_word[k] == 0) continue;
                    e[k] = c[k] * curr_word[k];
                    E += e[k];
                }



                // sample a new topic for the current word
                Q = E + F + G;
                double U = dis(gen) * Q;
                if(U < E){
                    topic = resample(e,U);
                }else if( U < E + F){
                    topic = resample(f, U - E);
                }else{
                    topic = resample(g, U - E - F);
                }

                // update all related values

                T[d][j] = topic;

                doc_topic_table[d][topic] ++;
                local_word_topic_table[word][topic] ++;
                local_topic_table[topic] ++;

                global_word_topic_table[word][topic] ++;
                global_topic_table[topic]++;

                denominator = global_topic_table[topic] + beta * vocab_size;
                F -= f[topic];
                f[topic] = (beta * curr_doc[topic]) / denominator;
                F += f[topic];
                c[topic] = (curr_doc[topic] + alpha) / denominator;
                G -= g[topic];
                g[topic] = (beta * alpha) / denominator;
                G += g[topic];

            }

        }
        double llh = getLocalLogLikelihood();

        double global_llh = 0;
        MPI_Allreduce(&llh, &global_llh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            global_llh += getGlobalLogLikelihood();
            LOG("Iteration: %d, loglikelihood: %.8f, time: %.2fs\n", iter, global_llh, timer.get_time_elapsed());
        }
        reduce_tables();
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

int lda::resample(std::vector<double> multi_dis, double prob) {

    double accum = 0;
    for(int i = 0; i < num_topics; i++){
        if(multi_dis[i] == 0) continue;
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

double lda::getGlobalLogLikelihood() {
    double lik = 0.0;
    for(int k = 0; k < num_topics; k++){
        for(int w = 0; w < vocab_size; w++){
            vocab_temp[w] = global_word_topic_table[w][k] + beta;
        }
        lik += logDirichlet(vocab_temp, vocab_size);
        lik -= logDirichlet(beta, vocab_size);
    }
    return lik;
}

double lda::getLocalLogLikelihood() {
    double lik = 0.0;


    for(int d = 0; d < num_docs; d++){
        std::unordered_map<int, int>& topic_vector = doc_topic_table[d];
        for(int k = 0; k < num_topics; k++){
            topic_temp[k] = topic_vector[k] + alpha;
        }
        lik += logDirichlet(topic_temp, num_topics);
        lik -= logDirichlet(alpha, num_topics);
    }

    return lik;
}

void lda::printTopicWord() {
    std::string fileName = "output/" + output + ".tw";
    std::ofstream out_file(fileName);
    for(int w = 0; w < vocab_size; w++){
        for(int k = 0; k < num_topics - 1; k++){
            out_file << global_word_topic_table[w][k] << ",";
        }
        out_file << global_word_topic_table[w][num_topics - 1] << "\n";
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
