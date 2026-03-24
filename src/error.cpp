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

#include "error.hpp"
#include <cstring>
#include <sstream>

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

} // namespace beancode::error
