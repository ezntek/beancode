#include <print>

#include "lexer.hpp"

using namespace beancode;
int main(int argc, char** argv) {
    argc--;
    argv++;

    if (argc == 0) {
        std::println(stderr, "not enough args");
        return 1;
    } else {
        std::println(stderr, "got: `{}`", *argv);
    }

    lexer::Lexer l(*argv);
    l.trim_comments();
    return 0;
}
