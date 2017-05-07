//
// Created by CHEN HU on 4/23/17.
//

#include <iostream>
#include <math.h>
#include <fstream>
#include <Log.h>
#include "lda.h"
#include "CycleTimer.h"


lda::lda(std::string dataDir, std::string output, int num_topics, double alpha, double beta, int num_iterations)
        : gen(std::random_device()()), dis(0, 1) {
    this->num_topics = num_topics;
    this->alpha = alpha;
    this->beta = beta;
    this->num_iterations = num_iterations;
    this->data_loader = new dataLoader(dataDir);
    this->output = output;

    num_docs = data_loader->docsCount();
    vocab_size = data_loader->vocabSize();

    topic_table = new int[num_topics]();
    topic_word_table = new int*[num_topics];
    for(int i = 0; i < num_topics; i++)
        topic_word_table[i] = new int[vocab_size]();

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
    delete[] topic_table;

    for(int i = 0; i < num_topics; i++)
        delete[] topic_word_table[i];
    delete[] topic_word_table;

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
            topic_word_table[topic][word] ++;
            topic_table[topic] ++;
        }
    }

}


void lda::runGibbs() {

    initialize();
    std::vector<double> dis((size_t)num_topics, 0);

    CycleTimer timer;
    for(int iter = 0; iter < num_iterations; iter++){
        for(int d = 0; d < (int) W.size(); d++){
            for(int j = 0; j < (int) W[d].size(); j++){
                int word = W[d][j];
                int topic = T[d][j];
                // ignore current position
                doc_topic_table[d][topic] --;
                topic_word_table[topic][word] --;
                topic_table[topic] --;

                // recalculate topic distribution
                for(int k = 0; k < num_topics; k++) {
                    dis[k] = (topic_word_table[k][word] + beta) / (topic_table[k] + beta * vocab_size) * (doc_topic_table[d][k] + alpha);
                }

                topic = resample(dis);
                T[d][j] = topic;
                doc_topic_table[d][topic] ++;
                topic_word_table[topic][word] ++;
                topic_table[topic] ++;

            }
        }
        double llh = getLogLikelihood();
        LOG("Iteration: %d, loglikelihood: %.8f, time: %.2fs\n", iter, llh, timer.get_time_elapsed());
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
        int* word_vector = topic_word_table[k];
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
            out_file << topic_word_table[k][w] << ",";
        }
        out_file << topic_word_table[k][vocab_size - 1] << "\n";
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
