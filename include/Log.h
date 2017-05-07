#ifndef PARALDA_LOG_H
#define PARALDA_LOG_H

#include <cstdio>
#include <cstdlib>


class Log {
    static FILE *log_file;
public:
    static void open(const char *path);
    static void log(const char *fmt, ...);
    static void close();
};


#define OPEN_LOG(path) Log::open(path)
#define LOG(...) Log::log(__VA_ARGS__)
#define CLOSE_LOG() Log::close()

#endif //PARALDA_LOG_H
