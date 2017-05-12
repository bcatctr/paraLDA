//
// Created by CHEN HU on 4/23/17.
//

#include <iostream>
#include <math.h>
#include <fstream>
#include "Log.h"
#include <thread>
//#include <omp.h>
#include "lda.h"
#include "CycleTimer.h"


lda::lda(std::string dataDir, std::string output, int num_topics,
         double alpha, double beta, int max_iterations, double threshold, int rank, int comm_size, int master_count, MPI_Comm MPI_COMM_WORKER)
        : gen(std::random_device()()), dis(0, 1) {
    LOG("start init lda\n");
    this->rank = rank;
    this->comm_size = comm_size;
    this->master_count = master_count;
    this->num_topics = num_topics;
    this->alpha = alpha;
    this->beta = beta;
    this->max_iterations = max_iterations;
    this->threshold = threshold;
    this->data_loader = new dataLoader(dataDir, rank, comm_size, master_count);
    this->output = output;
    this->MPI_COMM_WORKER = MPI_COMM_WORKER;


    communicator = new Communicator(master_count);

    num_docs = data_loader->docsCount();
    vocab_size = data_loader->vocabSize();

    memory_size = num_topics * vocab_size + num_topics;

    current = 0;

    local_table_memory[0] = new int[memory_size];
    local_table_memory[1] = new int[memory_size];
    global_table_memory[0] = new int[memory_size];
    global_table_memory[1] = new int[memory_size];

    local_topic_table[0] = local_table_memory[0] + num_topics * vocab_size;
    local_topic_table[1] = local_table_memory[1] + num_topics * vocab_size;
    global_topic_table[0] = global_table_memory[0] + num_topics * vocab_size;
    global_topic_table[1] = global_table_memory[1] + num_topics * vocab_size;

    local_word_topic_table[0] = new int*[vocab_size];
    local_word_topic_table[1] = new int*[vocab_size];
    global_word_topic_table[0] = new int*[vocab_size];
    global_word_topic_table[1] = new int*[vocab_size];
    int j = 0;
    for(int i = 0; i < vocab_size; i++) {
        local_word_topic_table[0][i] = local_table_memory[0] + j;
        local_word_topic_table[1][i] = local_table_memory[1] + j;
        global_word_topic_table[0][i] = global_table_memory[0] + j;
        global_word_topic_table[1][i] = global_table_memory[1] + j;
        j += num_topics;
    }
    memset(local_table_memory[0], 0, sizeof(int) * memory_size);
    memset(local_table_memory[1], 0, sizeof(int) * memory_size);
    memset(global_table_memory[0], 0, sizeof(int) * memory_size);
    memset(global_table_memory[1], 0, sizeof(int) * memory_size);

    doc_topic_table = new int*[num_docs];
    for(int i = 0; i < num_docs; i++)
        doc_topic_table[i] = new int[num_topics]();

    W = data_loader->loadCorpus();

    T.resize(W.size());
    for(int i = 0; i < (int)W.size(); i++){
        std::vector<int> temp(W[i].size(), -1);
        T[i] = temp;
    }

    vocab_temp = new double[vocab_size];
    topic_temp = new double[num_topics];

    LOG("finish init lda\n");
}

lda::~lda() {
    delete data_loader;

    delete communicator;

    delete[] global_table_memory[0];
    delete[] global_table_memory[1];
    delete[] local_table_memory[0];
    delete[] local_table_memory[1];

    delete[] global_word_topic_table[0];
    delete[] global_word_topic_table[1];
    delete[] local_word_topic_table[0];
    delete[] local_word_topic_table[1];

    for(int i = 0; i < num_docs; i++)
        delete[] doc_topic_table[i];
    delete[] doc_topic_table;

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
            local_word_topic_table[current][word][topic] ++;
            local_topic_table[current][topic] ++;
        }
    }
}

void lda::doIter() {


    memset(local_table_memory[current], 0, sizeof(int) * memory_size);

    for(int d = 0; d < (int) W.size(); d++){
        // calculate coefficient c which can be updated topic by topic in iterations by word

        // calculate "doc-topic" bucket F and coefficient c
        F = 0;
        std::fill(f.begin(),f.end(),0);
        int* curr_doc = doc_topic_table[d];
        for(int k = 0; k < num_topics ; k++){
            denominator = global_topic_table[current][k] + beta * vocab_size;
            c[k] = (curr_doc[k] + alpha) / denominator;
            if(curr_doc[k] == 0) continue;

            f[k] = (beta * curr_doc[k]) / denominator;
            F += f[k];
        }

        // sample a new topic for each word of the document
        for(int j = 0; j < (int) W[d].size(); j++){
            int word = W[d][j];
            int topic = T[d][j];
            // ignore current position, update all related value
            doc_topic_table[d][topic] --;
            local_word_topic_table[current][word][topic] --;
            local_topic_table[current][topic] --;

            global_word_topic_table[current][word][topic] --;
            global_topic_table[current][topic]--;

            denominator = global_topic_table[current][topic] + beta * vocab_size;
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
            int* curr_word = global_word_topic_table[current][word];
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
            local_word_topic_table[current][word][topic] ++;
            local_topic_table[current][topic] ++;

            global_word_topic_table[current][word][topic] ++;
            global_topic_table[current][topic]++;

            denominator = global_topic_table[current][topic] + beta * vocab_size;
            F -= f[topic];
            f[topic] = (beta * curr_doc[topic]) / denominator;
            F += f[topic];
            c[topic] = (curr_doc[topic] + alpha) / denominator;
            G -= g[topic];
            g[topic] = (beta * alpha) / denominator;
            G += g[topic];

        }

    }
}

void lda::runGibbs() {

    initialize();

    c.resize((size_t)num_topics, 0);
    f.resize((size_t)num_topics, 0);
    g.resize((size_t)num_topics, 0);
    e.resize((size_t)num_topics, 0);

    bool flag = true;
    double elapsed_time = 0;
    double iteration_time = 0;
    double old_llh = 0;

    //precompute the likelihood
    double llh = getLocalLogLikelihood();
    MPI_Allreduce(&llh, &old_llh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORKER);
    old_llh += getGlobalLogLikelihood();


    CycleTimer timer;

    LOG("start sync\n");
    // blocking communication only once
    communicator->ISend(local_table_memory[current], memory_size);
    communicator->Recv(global_table_memory[current], memory_size);
    LOG("finish sync\n");

    // pre-calculate G and g because they are irrelevant with document
    G = 0;
    for(int k = 0; k < num_topics; k++){
        g[k] = beta * alpha / (global_topic_table[current][k] + beta * vocab_size);
        G += g[k];
    }

    LOG("start iterations\n");

    for(int iter = 0; flag && iter < max_iterations; iter++){

        CycleTimer iter_timer;

        std::thread t([this]() {
            doIter();
        });

        if (iter != 0) {
            communicator->Wait();
        }
        t.join();

        iteration_time = iter_timer.get_time_elapsed();
        elapsed_time += iteration_time;

        double global_llh = 0;
        //MPI_Allreduce(&llh, &global_llh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORKER);

        global_llh += getGlobalLogLikelihood();

        double diff = std::abs(old_llh - global_llh) / std::abs(global_llh);
        old_llh = global_llh;
        flag = (diff > threshold);

        if (rank == master_count) {

            LOG("Iteration:%d\tloglikelihood:%.8f\trel_change:%.4f\ttime:%.4f\ttotal_time:%.2f\n", iter, global_llh, diff, iteration_time, elapsed_time);
        }

        if (iter != 0) {
            communicator->ISend(local_table_memory[current], memory_size);
            communicator->IRecv(global_table_memory[current], memory_size);
            current = 1 - current;
        }
        else {
            memcpy(local_table_memory[1 - current], local_table_memory[current], memory_size * sizeof(int));
            communicator->ISend(local_table_memory[1 - current], memory_size);
            communicator->IRecv(global_table_memory[1 - current], memory_size);
        }
    }
    communicator->Wait();

    communicator->Complete();
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
            vocab_temp[w] = global_word_topic_table[current][w][k] + beta;
        }
        lik += logDirichlet(vocab_temp, vocab_size);
        lik -= logDirichlet(beta, vocab_size);
    }
    return lik;
}

double lda::getLocalLogLikelihood() {
    double lik = 0.0;


    for(int d = 0; d < num_docs; d++){
        int* topic_vector = doc_topic_table[d];
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
            out_file << global_word_topic_table[current][w][k] << ",";
        }
        out_file << global_word_topic_table[current][w][num_topics - 1] << "\n";
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
