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
#pragma once

#include <exception>
#include <string>

#include "lexer.hpp"

namespace beancode::error {

class BCError : public std::exception {
public:
    enum class Kind : u8 {
        Runtime,
        Syntax,
        Eof,
    };

    BCError(Kind k, lexer::Pos p);
    BCError(Kind k, lexer::Pos p, std::string msg);
    BCError(Kind k, lexer::Pos p, std::string msg, std::string context);

    std::optional<std::string> context;
    std::string msg;
    lexer::Pos pos;
    Kind kind;

    static std::string kind_to_string(const Kind& k);
    const char* what() const noexcept;
    std::string to_string() const noexcept;
};

} // namespace beancode::error
