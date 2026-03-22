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

#include <format>
#include <string>

#include "./ast.hpp"
#include "./common.hpp"

namespace beancode::lexer {

struct Pos {
    u32 row;
    u32 col;
    u32 span;

    std::string to_string() const;
};

class Lexer {
private:
    std::string src;
    u64 cur;
    u32 row;
    u32 col;

    inline char32_t get_cur_cp() const;
    inline bool in_bounds() const;
    Pos pos(u32 span) const;
    Pos pos_here(u32 span) const;
    inline bool is_separator(char c) const;
    inline bool is_operator_start(char* c) const;

    inline void next();
    inline void next(u8 count);
    void bump_newline();

public:
    Lexer(std::string src);
    void reset();
    void trim_spaces();
    void trim_comments();
};

}; // namespace beancode::lexer
