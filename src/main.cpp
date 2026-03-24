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

    auto file_name = *argv;

    if (argc == 0) {
        std::println(stderr, "not enough args");
        return 1;
    }

    std::ifstream ifs{file_name};
    std::stringstream ss;

    if (!ifs.is_open()) {
        std::println(stderr, "could not open file {}", file_name);
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
        e.print(file_name);
        return 1;
    }

    return 0;
}
