#include <print>

#include "error.hpp"
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

    std::string src = *argv;
    lexer::Lexer l(src);
    try {
        auto tokens = l.tokenize();
        for (const auto& tok : tokens)
            tok.print();
    } catch (error::BCError& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }

    return 0;
}
