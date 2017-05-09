#include <iostream>
#include <unordered_map>
#include <fstream>
#include "lda.h"
#include "Log.h"
#include "CycleTimer.h"
#include "mpi.h"

std::unordered_map<std::string, std::string> parseArg(char *fileName);

int main(int argc, char *argv[]) {

    int rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if(argc <= 1){
        printf("Missing parameter file path!\n");
        return 1;
    }
    std::unordered_map<std::string, std::string> paras = parseArg(argv[1]);
    if (paras.count("logFile")) {
        OPEN_LOG(paras["logFile"].c_str());
    }
    else {
        OPEN_LOG("");
    }

    CycleTimer timer;
    LOG("start main\n");

    lda my_lda(paras["dataPath"],
               paras["outputFile"],
               std::stoi(paras["numTopics"]),
               std::stod(paras["alpha"]),
               std::stod(paras["beta"]),
               std::stoi(paras["numIterations"]),
               rank,
               comm_size);
    my_lda.runGibbs();

    if (rank == 0) {
        my_lda.printTopicWord();
        my_lda.printDocTopic();
    }

    LOG("finish main, %.2fs\n", timer.get_time_elapsed());
    CLOSE_LOG();

    MPI_Finalize();

    return 0;
}

std::unordered_map<std::string, std::string> parseArg(char *fileName){
    std::ifstream paraFile(fileName);

    std::string line;
    std::unordered_map<std::string, std::string> paras;


    while(std::getline(paraFile,line)){
        size_t pos = line.find_first_of(':');
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        paras[key] = value;
    }

    paraFile.close();

    return paras;
}



