#include <iostream>
#include <unordered_map>
#include <fstream>
#include "lda.h"

std::unordered_map<std::string, std::string> parseArg(char *fileName);

int main(int argc, char *argv[]) {

    if(argc <= 1){
        std::cout << "Missing parameter file path!" << std::endl;
        return 1;
    }
    std::unordered_map<std::string, std::string> paras = parseArg(argv[1]);
    lda my_lda(paras["dataDirectory"],std::stoi(paras["numTopics"]),std::stod(paras["alpha"]),std::stod(paras["beta"]),std::stoi(paras["numIterations"]));
    my_lda.runGibbs();

    return 0;
}

std::unordered_map<std::string, std::string> parseArg(char *fileName){
    std::ifstream paraFile("parameters.txt");

    std::string line;
    std::unordered_map<std::string, std::string> paras;


    while(std::getline(paraFile,line)){
        int pos = line.find_first_of(':');
        std::string key = line.substr(0,pos);
        std::string value = line.substr(pos + 1);
        paras[key] = value;
    }

    return paras;
}



