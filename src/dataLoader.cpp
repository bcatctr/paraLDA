//
// Created by CHEN HU on 4/23/17.
//

#include "dataLoader.h"
#include "Log.h"
#include "CycleTimer.h"
#include <fstream>
#include <sstream>

dataLoader::dataLoader(std::string dataDir, int rank, int comm_size, int master_count) {
    std::string dictFile = dataDir + ".dict";
    std::string dataFile = dataDir + ".data";

    CycleTimer timer;

    LOG("start loading dictionary: %s\n", dictFile.c_str());
    std::ifstream dict_file(dictFile);
    std::string line;
    while(std::getline(dict_file, line)){
        this->dict.push_back(line);
    }
    LOG("dictionary size: %d\n", this->dict.size());
    LOG("finish loading dictionary, %.2fs\n", timer.get_time_elapsed());

    if (rank < master_count) return;

    LOG("start loading data: %s\n", dataFile.c_str());

    comm_size -= master_count;
    rank -= master_count;

    int line_idx = 0;
    std::ifstream data_file(dataFile);
    while(std::getline(data_file, line)){
        if (line_idx++ % comm_size == rank) {
            std::vector<int> doc;
            std::stringstream ss(line);
            std::string tok;

            while(std::getline(ss, tok, ',')){
                doc.push_back(std::stoi(tok));
            }

            this->corpus.push_back(doc);
        }
    }
    data_file.close();

    LOG("data size: %d\n", this->corpus.size());
    LOG("finish loading data, %.2fs\n", timer.get_time_elapsed());

}

dataLoader::~dataLoader() {
    this->dict.clear();
    this->corpus.clear();
}

int dataLoader::vocabSize() {
    return (int) this->dict.size();
}

int dataLoader::docsCount() {
    return (int) this->corpus.size();
}

std::vector<std::string> dataLoader::loadDict() {
    return this->dict;
}

std::vector<std::vector<int>> dataLoader::loadCorpus() {
    return this->corpus;
}