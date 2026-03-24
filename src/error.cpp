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

#include <print>
#include <sstream>

#include "error.hpp"

namespace beancode::error {

BCError::BCError(Kind k, lexer::Pos p) : pos(p), kind(k) {
    msg = kind_to_string(k);
}

BCError::BCError(Kind k, lexer::Pos p, std::string msg) : msg(msg), pos(p), kind(k) {
}

BCError::BCError(Kind k, lexer::Pos p, std::string msg, std::string context)
    : context(context), msg(msg), pos(p), kind(k) {
}

std::string BCError::kind_to_string(const Kind& k) {
    switch (k) {
        case Kind::Eof: {
            return "EOFError";
        } break;
        case Kind::Syntax: {
            return "SyntaxError";
        } break;
        case Kind::Runtime: {
            return "RuntimeError";
        } break;
    }
}

const char* BCError::what() const noexcept {
    return msg.c_str();
}

std::string BCError::to_string() const noexcept {
    std::stringstream ss;
    std::string k = kind_to_string(kind);

    if (msg == k) {
        return k;
    }

    ss << k << ": " << msg;

    if (context) {
        ss << " (context: " << context.value() << ")";
    }

    return ss.str();
}

// TODO: write a better impl that is not a direct rewrite from the Python codebase
void BCError::print(const std::string_view file_name, FILE* f, bool color) const noexcept {
    if (color) {
        std::print(f, "\033[1m{}:{}: \033[31;1merror: \x1b[0m", file_name, pos.row);
    } else {
        std::print(f, "{}:{}: error: ", file_name, pos.row);
    }

    std::print(f, "{}", msg);

    if (context.has_value()) {
        if (color) {
            std::print(f, "\n\033[2m");
        } else {
            std::println(f);
        }

        // "error: " + file_name + ":" + pos.row + ":"
        for (usize i = 0; i < 9 + file_name.length(); ++i)
            std::print(f, " ");

        for (auto tmp = pos.row; tmp; tmp /= 10)
            std::print(f, " ");

        std::print(f, "{}", context.value());

        if (color) std::print("\033[0m");
    }

    std::println(f);
}

void BCError::print(const std::string_view file_name, const std::string_view src, FILE* f, bool color) const noexcept {
    // print the header first
    print(file_name, f, color);

    // TODO: source code print

    (void)src;
}

} // namespace beancode::error
