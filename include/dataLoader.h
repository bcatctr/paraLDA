//
// Created by CHEN HU on 4/23/17.
//

#ifndef LDA_DATALOADER_H
#define LDA_DATALOADER_H


#include <string>
#include <vector>

class dataLoader {

    std::string dataFile;
    std::vector<std::string> dict;
    std::vector<std::vector<int>> corpus;

public:
    dataLoader(std::string dataDir);

    ~dataLoader();

    int docsCount();

    int volcabSize();

    std::vector<std::string> loadDict();

    std::vector<std::vector<int>> loadCorpus();

};


#endif //LDA_DATALOADER_H
