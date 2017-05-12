#include <iostream>
#include <unordered_map>
#include <fstream>
#include "Master.h"
#include "lda.h"
#include "Log.h"
#include "CycleTimer.h"

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

    int master_count = std::stoi(paras["masterCount"]);
    LOG("master count: %d\n", master_count);

    LOG("start create comm\n");
    MPI_Group MPI_GROUP_WORLD;
    MPI_Group MPI_GROUP_WORKER;
    MPI_Comm MPI_COMM_WORKER;
    int* master_ranks = new int[master_count];
    for (int i = 0; i < master_count; ++i) {
        master_ranks[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
    MPI_Group_excl(MPI_GROUP_WORLD, master_count, master_ranks, &MPI_GROUP_WORKER);
    LOG("succes?: %d\n", MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_WORKER, &MPI_COMM_WORKER) == MPI_SUCCESS);
    LOG("comm worker: %d\n", MPI_COMM_WORKER);
    delete[] master_ranks;
    LOG("finish create comm\n");

    CycleTimer timer;
    LOG("start main\n");

    if (rank >= master_count) {
        // run worker
        lda my_lda(paras["dataPath"],
                   paras["outputFile"],
                   std::stoi(paras["numTopics"]),
                   std::stod(paras["alpha"]),
                   std::stod(paras["beta"]),
                   std::stoi(paras["numIterations"]),
                   rank,
                   comm_size,
                   master_count,
                   MPI_COMM_WORKER);
        my_lda.runGibbs();

        if (rank == master_count) {
            my_lda.printTopicWord();
            my_lda.printDocTopic();
        }
    }
    else {
        // run master
        dataLoader d(paras["dataPath"], rank, comm_size, master_count);
        int num_topics = std::stoi(paras["numTopics"]);
        int length = num_topics * d.vocabSize() + num_topics;
        Master master(length, comm_size);
        master.run();
    }

    LOG("rank: %d, finish main, %.2fs\n", rank, timer.get_time_elapsed());
    CLOSE_LOG();

    MPI_Barrier(MPI_COMM_WORLD);
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



