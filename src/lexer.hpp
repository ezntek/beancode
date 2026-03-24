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

#include <optional>
#include <string>
#include <vector>

#include "./ast.hpp"
#include "./common.hpp"

namespace beancode::lexer {

// TODO: maybe a file ID for multifile?
struct Pos {
    u32 row;
    u16 col;
    u16 span;

    std::string to_string() const;
};

struct Token {
    enum class Kind : u8 {
        Bogus = 0,
        Declare,
        Constant,
        Output,
        Input,
        And,
        Or,
        Not,
        If,
        Then,
        Else,
        Endif,
        Case,
        Of,
        Otherwise,
        Endcase,
        While,
        Do,
        Endwhile,
        Repeat,
        Until,
        For,
        To,
        Step,
        Next,
        Procedure,
        Endprocedure,
        Call,
        Function,
        Return,
        Returns,
        Endfunction,
        Openfile,
        Readfile,
        Writefile,
        Closefile,
        Read,
        Write,
        Append,
        Include,
        IncludeFfi,
        Export,
        Scope,
        Endscope,
        Print,
        Trace,
        Endtrace,
        Assign,
        Equal,
        LessThan,
        GreaterThan,
        LessThanOrEqual,
        GreaterThanOrEqual,
        NotEqual,
        Mul,
        Div,
        Add,
        Sub,
        Pow,
        LeftParen,
        RightParen,
        LeftBracket,
        RightBracket,
        LeftCurly,
        RightCurly,
        Colon,
        Comma,
        Dot,
        Newline,
        LiteralString,
        LiteralChar,
        LiteralNumber,
        True,
        False,
        Null,
        Ident,
        TInteger,
        TBoolean,
        TReal,
        TChar,
        TString,
        TArray,
        // TODO: CST node for formatter
        Comment,
    };

    std::string_view data;
    Pos pos;
    Kind kind;

    Token();
    Token(Kind k, Pos p);
    Token(Kind k, Pos p, std::string_view s);

    static std::string kind_to_string(const Kind& k);
    std::string to_string() const;
    void print(FILE* f = stdout) const;
};

class Lexer {
private:
    std::string_view src;
    u64 cur;
    u64 bol;
    u32 row;

    inline char32_t get_cur_cp() const;
    inline bool in_bounds() const;
    Pos pos(u16 span) const;
    Pos pos_here(u16 span) const;
    inline bool is_separator(char c) const;
    inline bool is_operator_start(const char* c) const;

    inline void next();
    inline void next(u8 count);
    void bump_newline();

    std::string_view next_word();

    auto next_multi_symbol() -> std::optional<Token>;
    auto next_single_symbol() -> std::optional<Token>;
    auto next_keyword(const std::string_view word) -> std::optional<Token>;
    auto next_type(const std::string_view word) -> std::optional<Token>;
    auto next_literal(const std::string_view word) -> std::optional<Token>;
    auto next_ident(const std::string_view word) -> std::optional<Token>;

public:
    Lexer(const std::string& src);
    void reset();
    void trim_spaces();
    void trim_comments();

    auto next_token() -> std::optional<Token>;
    auto tokenize() -> std::vector<Token>;
};

}; // namespace beancode::lexer
