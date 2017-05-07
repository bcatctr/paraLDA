#include "Log.h"
#include <cstdarg>
#include <cstring>

FILE *Log::log_file = nullptr;

void Log::open(const char *path) {
    if (strlen(path) == 0) {
        log_file = fopen(path, "w");
    }
    else {
        log_file = nullptr;
    }
}

void Log::log(const char *fmt, ...) {
    va_list arg;
    va_start(arg, fmt);
    vfprintf(stdout, fmt, arg);
    fflush(stdout);

    if (log_file != nullptr) {
        vfprintf(log_file, fmt, arg);
        fflush(log_file);
    }

    va_end(arg);
}

void Log::close() {
    if (log_file != nullptr) {
        fclose(log_file);
    }
}
