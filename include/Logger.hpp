#pragma once

#include <string>
#include <sstream>
#include <iostream>

class Logger {
public:
    enum Importance {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };
    static void log(Importance importance, std::string const &message, std::string const &file, int line) {
        const std::string prefixes[] = {
            "\033[1;37m [DEBUG]",
            "\033[1;32m [INFO]",
            "\033[1;33m [WARNING]",
            "\033[1;31m [ERROR]"
        };
        const auto reset = "\033[0m ";

        std::cerr << prefixes[(int)importance] << "[\"" << file << "\" (line " << line <<")] " << message << reset << std::endl;
    }
};

#define LOG(importance, message) do {Logger::log(importance, static_cast<std::ostringstream&>(\
      std::ostringstream().flush() << message  \
    ).str(), __FILE__, __LINE__);} while (0)
