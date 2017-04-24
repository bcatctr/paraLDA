//
// Created by CHEN HU on 4/23/17.
//

#include <random>
#include "lda.h"

lda::lda(std::string dataFile, int num_topics, double alpha, double beta, int num_iterations): gen(std::random_device()()), dis(0,1){
    this->num_topics = num_topics;
    this->alpha = alpha;
    this->beta = beta;
    this->num_iterations = num_iterations;
    this->data_loader = new dataLoader(dataFile);

    this->num_docs = this->data_loader->docsCount();
    this->vocab_size = this->data_loader->volcabSize();

    this->topic_table = new int[this->num_topics]();
    this->topic_word_table = new int*[this->num_topics];
    for(int i = 0; i < this->num_topics; i++)
        this->topic_word_table[i] = new int[this->vocab_size]();

    this->doc_topic_table = new int*[this->num_docs];
    for(int i = 0; i < this->num_docs; i++)
        this->doc_topic_table[i] = new int[this->num_topics]();

    this->W = this->data_loader->loadCorpus();

    this->T.resize(this->W.size());
    for(int i = 0; i < this->W.size(); i++){
        this->T.at(i) = new std::vector<int>(this->W.at(i)->size(), -1);
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

    for(int i = 0; i < this->T.size(); i++)
        delete this->T.at(i);
    this->T.clear();

}

void lda::initialize() {
    std::random_device rd;
    std::mt19937 int_gen(rd());
    std::uniform_int_distribution<> int_dis(0,this->num_topics - 1);

    for(int d = 0; d < this->T.size(); d++){
        for(int j = 0; j < this->T.at(d)->size(); j++){
            int word = this->W.at(d)->at(j);
            int topic = int_dis(int_gen);
            this->T.at(d)->at(j) = topic;
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
        for(int d = 0; d < this->W.size(); d++){
            for(int j = 0; j < this->W.at(d)->size(); j++){
                int word = this->W.at(d)->at(j);
                int topic = this->T.at(d)->at(j);
                // ignore current position
                this->doc_topic_table[d][topic] --;
                this->topic_word_table[topic][word] --;
                this->topic_table[topic] --;

                // recalculate topic distribution
                for(int k = 0; k < this->num_topics; k++){
                    dis.at(k) = (this->topic_word_table[k][word] + this->beta)/(double)(this->topic_table[j] + this->beta * this->vocab_size) * (this->doc_topic_table[d][k] + this->alpha);
                }

                topic = this->resample(dis);
                this->T.at(d)->at(j) = topic;
                this->doc_topic_table[d][topic] ++;
                this->topic_word_table[topic][word] ++;
                this->topic_table[topic] ++;

            }
        }
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
