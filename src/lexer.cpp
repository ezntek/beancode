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

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <format>
#include <string>
#include <string_view>

#include "common.hpp"
#include "lexer.hpp"
#include "utf8.hpp"

namespace beancode::lexer {

Lexer::Lexer(std::string src) : src(src), row(0), col(0) {
}

std::string Pos::to_string() const {
    return std::format("{} {} {}", row, col, span);
}

inline void Lexer::next() {
    ++col;
    static auto e = src.end();
}

inline void Lexer::next(u8 count) {
    static auto e = src.end();
}

inline char32_t Lexer::get_cur_cp() const {
    char32_t res;
    assert(res = utf8::decode(&src[cur]) >= 0);
    return res;
}

inline bool Lexer::in_bounds() const {
    return cur < src.length();
}

Pos Lexer::pos(u32 span) const {
    return Pos{row, col - span, span};
}

Pos Lexer::pos_here(u32 span) const {
    return Pos{row, col, span};
}

inline bool Lexer::is_separator(char ch) const {
    return strchr("[]{}();:.", ch) != NULL;
}

inline bool Lexer::is_operator_start(char* ch) const {
    if (strchr("%+-*/<>=^", *ch) != NULL)
        return true;

    int32_t res = utf8::decode(ch);
    assert(res >= 0);

    // unicode ←
    switch (res) {
        case 0x2190:
        case 0xf0ac:
            return true;
        default:
            return false;
    }
}

void Lexer::bump_newline() {
    cur++;
    row++;
    col = 1;
}

void Lexer::reset() {
    row = 1;
    col = 0;
    cur = 0;
}

void Lexer::trim_spaces() {
    if (!in_bounds())
        return;

    char ch;
    while (in_bounds() && isspace(src[cur]) && src[cur] != '\n')
        next();

    trim_comments();
}

void Lexer::trim_comments() {
    std::string_view src_view = src, pair;

    if (cur + 2 > src.length())
        return;

    pair = src_view.substr(cur, cur + 2);
    if (pair == "/*") {
        cur += 2;

        while (in_bounds() && src_view.substr(cur, cur + 2) != "*/") {
            if (src[cur] == '\n')
                bump_newline();
            else
                cur++;
        }

        // found the delimiter
        cur++;
    } else if (pair == "#!" || pair == "//") {
        cur += 2;
        while (in_bounds() && src[cur] != '\n')
            cur++;
        // don't skip the newline, the whitespace function will do it
    }

    trim_spaces();
}

}; // namespace beancode::lexer

template <>
struct std::formatter<beancode::lexer::Pos> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const beancode::lexer::Pos& p, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", p.to_string());
    }
};
