/*
 * beancode: a portable IGCSE Computer Science (0478, 0984, 2210) Pseudocode
 * interpreter.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <fstream>
#include <print>
#include <sstream>

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

    std::ifstream ifs(*argv);
    std::stringstream ss;

    if (!ifs.is_open()) {
        std::println(stderr, "could not open file {}", *argv);
        return 1;
    }

    ss << ifs.rdbuf();
    auto src = ss.str();

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
