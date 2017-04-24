//
// Created by CHEN HU on 4/23/17.
//

#ifndef LDA_DATALOADER_H
#define LDA_DATALOADER_H


#include <string>
#include <vector>

class dataLoader {

    std::string dataFile;



public:
    dataLoader(std::string dataFile){
        this->dataFile = dataFile;
    }

    ~dataLoader();

    int docsCount();

    int volcabSize();

    std::vector<std::vector<int>*> loadCorpus();

};


#endif //LDA_DATALOADER_H
