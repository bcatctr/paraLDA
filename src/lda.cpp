//
// Created by CHEN HU on 4/23/17.
//

#include <iostream>
#include <math.h>
#include <fstream>
#include "lda.h"

lda::lda(std::string dataDir, std::string output, int num_topics, double alpha, double beta, int num_iterations): gen(std::random_device()()), dis(0,1){
    this->num_topics = num_topics;
    this->alpha = alpha;
    this->beta = beta;
    this->num_iterations = num_iterations;
    this->data_loader = new dataLoader(dataDir);
    this->output = output;

    this->num_docs = this->data_loader->docsCount();
    this->vocab_size = this->data_loader->volcabSize();

    this->topic_table = new int[this->num_topics]();
    this->topic_word_table = new int*[this->num_topics];
    for(int i = 0; i < (int)this->num_topics; i++)
        this->topic_word_table[i] = new int[this->vocab_size]();

    this->doc_topic_table = new int*[this->num_docs];
    for(int i = 0; i < (int) this->num_docs; i++)
        this->doc_topic_table[i] = new int[this->num_topics]();

    this->W = this->data_loader->loadCorpus();

    this->T.resize(this->W.size());
    for(int i = 0; i < (int)this->W.size(); i++){
        std::vector<int> temp(this->W.at(i).size(), -1);
        this->T.at(i) = temp;
    }


}

lda::~lda() {
    delete this->data_loader;
    delete[] this->topic_table;

    for(int i = 0; i < this->num_topics; i++)
        delete[] this->topic_word_table[i];
    delete[] this->topic_word_table;

    for(int i = 0; i < this->num_docs; i++)
        delete[] this->doc_topic_table[i];
    delete[] this->doc_topic_table;

    this->T.clear();
}

void lda::initialize() {
    std::random_device rd;
    std::mt19937 int_gen(rd());
    std::uniform_int_distribution<> int_dis(0,this->num_topics - 1);

    for(int d = 0; d < (int) this->T.size(); d++){
        for(int j = 0; j < (int) this->T.at(d).size(); j++){
            int word = this->W.at(d).at(j);
            int topic = int_dis(int_gen);
            this->T.at(d).at(j) = topic;
            this->doc_topic_table[d][topic] ++;
            this->topic_word_table[topic][word] ++;
            this->topic_table[topic] ++;
        }
    }

}


void lda::runGibbs() {

    this->initialize();
    std::vector<double> dis(this->num_topics,0);

    for(int iter = 0; iter < this->num_iterations; iter++){
        for(int d = 0; d < (int) this->W.size(); d++){
            for(int j = 0; j < (int) this->W.at(d).size(); j++){
                int word = this->W.at(d).at(j);
                int topic = this->T.at(d).at(j);
                // ignore current position
                this->doc_topic_table[d][topic] --;
                this->topic_word_table[topic][word] --;
                this->topic_table[topic] --;

                // recalculate topic distribution
                for(int k = 0; k < this->num_topics; k++){
                    dis.at(k) = (this->topic_word_table[k][word] + this->beta)/(double)(this->topic_table[k] + this->beta * this->vocab_size) * (this->doc_topic_table[d][k] + this->alpha);
                }

                topic = this->resample(dis);
                this->T.at(d).at(j) = topic;
                this->doc_topic_table[d][topic] ++;
                this->topic_word_table[topic][word] ++;
                this->topic_table[topic] ++;

            }
        }
        double llh = this->getLogLikelihood();
        std::cout << "Iteration "<< iter << ": " << llh << std::endl;
    }
}


int lda::resample(std::vector<double> multi_dis) {

    // normalize
    double sum = 0;
    for(int i = 0; i < this->num_topics; i++){
        sum += multi_dis.at(i);
    }


    double prob = this->dis(this->gen)*sum;

    double accum = 0;
    for(int i = 0; i < this->num_topics; i++){
        accum += multi_dis.at(i);
        if(prob < accum)
            return i;
    }
    return this->num_topics - 1;

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

    double* temp = new double[this->vocab_size];
    for(int k = 0; k < this->num_topics; k++){
        int* word_vector = this->topic_word_table[k];
        for(int w = 0; w < this->vocab_size; w++){
            temp[w] = word_vector[w] + this->beta;
        }
        lik += this->logDirichlet(temp, this->vocab_size);
        lik -= this->logDirichlet(this->beta, this->vocab_size);
    }
    delete[] temp;

    temp = new double[this->num_topics];
    for(int d = 0; d < this->num_docs; d++){
        int* topic_vector = this->doc_topic_table[d];
        for(int k = 0; k < this->num_topics; k++){
            temp[k] = topic_vector[k] + this->alpha;
        }
        lik += this->logDirichlet(temp, this->num_topics);
        lik -= this->logDirichlet(this->alpha, this->num_topics);
    }
    delete[] temp;

    return lik;
}

void lda::printTopicWord() {
    std::string fileName = "Output/" + this->output + ".tw";
    std::ofstream out_file(fileName);
    for(int k = 0; k < this->num_topics; k++){
        for(int w = 0; w < this->vocab_size - 1; w++){
            out_file << this->topic_word_table[k][w] << ",";
        }
        out_file << this->topic_word_table[k][this->vocab_size - 1] << "\n";
    }
}

void lda::printDocTopic() {
    std::string fileName = "Output/" + this->output + ".dt";
    std::ofstream out_file(fileName);
    for(int d = 0; d < this->num_docs; d++){
        for(int k = 0; k < this->num_topics - 1; k++){
            out_file << this->doc_topic_table[d][k] << ",";
        }
        out_file << this->doc_topic_table[d][this->num_topics - 1] << "\n";
    }

}
