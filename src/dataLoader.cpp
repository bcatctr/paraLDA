//
// Created by CHEN HU on 4/23/17.
//

#include "dataLoader.h"
#include <fstream>
#include <sstream>

dataLoader::dataLoader(std::string dataDir) {
    std::string dataFile = dataDir + "/" + dataDir + ".data";
    std::string dictFile = dataDir + "/" + dataDir + ".dict";

    std::ifstream dict_file(dictFile);
    std::string line;
    while(std::getline(dict_file, line)){
        this->dict.push_back(line);
    }

    std::ifstream data_file(dataFile);
    while(std::getline(data_file, line)){
        std::vector<int> doc;
        std::stringstream ss(line);
        std::string tok;

        while(std::getline(ss, tok, ',')){
            doc.push_back(std::stoi(tok));
        }

        this->corpus.push_back(doc);
    }

}

dataLoader::~dataLoader() {
    this->dict.clear();
    this->corpus.clear();
}

int dataLoader::volcabSize() {
    return this->dict.size();
}

int dataLoader::docsCount() {
    return this->corpus.size();
}

std::vector<std::string> dataLoader::loadDict() {
    return this->dict;
}

std::vector<std::vector<int>> dataLoader::loadCorpus() {
    return this->corpus;
}