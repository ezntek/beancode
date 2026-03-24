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

#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common.hpp"
#include "error.hpp"
#include "lexer.hpp"
#include "utf8.hpp"
#include "util.hpp"

namespace beancode::lexer {

#define CUR (src[cur])

using namespace error;

static Token::Kind token_kind_from_single_op(char ch) {
    using K = Token::Kind;

    switch (ch) {
        case '{': return K::LeftCurly;
        case '}': return K::RightCurly;
        case '[': return K::LeftBracket;
        case ']': return K::RightBracket;
        case '(': return K::LeftParen;
        case ')': return K::RightParen;
        case ':': return K::Colon;
        case ';': return K::Newline;
        case ',': return K::Comma;
        case '=': return K::Equal;
        case '<': return K::LessThan;
        case '>': return K::GreaterThan;
        case '*': return K::Mul;
        case '/': return K::Div;
        case '+': return K::Add;
        case '-': return K::Sub;
        case '^': return K::Pow;
        default: return K::Bogus;
    }
}

static Token::Kind token_kind_from_multi_op(const std::string_view s) {
    using K = Token::Kind;
    using namespace util;

    // U+2190 ←
    if (case_compare(s, "\xe2\x86\x90")) return K::Assign;
    // U+F0AC
    if (case_compare(s, "\xef\x82\xac")) return K::Assign;
    if (case_compare(s, "<-")) return K::Assign;
    if (case_compare(s, "<>")) return K::NotEqual;
    if (case_compare(s, ">=")) return K::GreaterThanOrEqual;
    if (case_compare(s, "<=")) return K::LessThanOrEqual;

    return K::Bogus;
}

static Token::Kind token_kind_from_keyword(const std::string_view s) {
    using K = Token::Kind;
    using namespace util;

    if (case_compare(s, "declare")) return K::Declare;
    if (case_compare(s, "constant")) return K::Constant;
    if (case_compare(s, "output")) return K::Output;
    if (case_compare(s, "input")) return K::Input;
    if (case_compare(s, "and")) return K::And;
    if (case_compare(s, "or")) return K::Or;
    if (case_compare(s, "not")) return K::Not;
    if (case_compare(s, "if")) return K::If;
    if (case_compare(s, "then")) return K::Then;
    if (case_compare(s, "else")) return K::Else;
    if (case_compare(s, "endif")) return K::Endif;
    if (case_compare(s, "case")) return K::Case;
    if (case_compare(s, "of")) return K::Of;
    if (case_compare(s, "otherwise")) return K::Otherwise;
    if (case_compare(s, "endcase")) return K::Endcase;
    if (case_compare(s, "while")) return K::While;
    if (case_compare(s, "do")) return K::Do;
    if (case_compare(s, "endwhile")) return K::Endwhile;
    if (case_compare(s, "repeat")) return K::Repeat;
    if (case_compare(s, "until")) return K::Until;
    if (case_compare(s, "for")) return K::For;
    if (case_compare(s, "to")) return K::To;
    if (case_compare(s, "step")) return K::Step;
    if (case_compare(s, "next")) return K::Next;
    if (case_compare(s, "procedure")) return K::Procedure;
    if (case_compare(s, "endprocedure")) return K::Endprocedure;
    if (case_compare(s, "call")) return K::Call;
    if (case_compare(s, "function")) return K::Function;
    if (case_compare(s, "returns")) return K::Returns;
    if (case_compare(s, "return")) return K::Return;
    if (case_compare(s, "endfunction")) return K::Endfunction;
    if (case_compare(s, "openfile")) return K::Openfile;
    if (case_compare(s, "readfile")) return K::Readfile;
    if (case_compare(s, "writefile")) return K::Writefile;
    if (case_compare(s, "closefile")) return K::Closefile;
    if (case_compare(s, "read")) return K::Read;
    if (case_compare(s, "write")) return K::Write;
    if (case_compare(s, "append")) return K::Append;
    if (case_compare(s, "trace")) return K::Trace;
    if (case_compare(s, "endtrace")) return K::Endtrace;
    if (case_compare(s, "scope")) return K::Scope;
    if (case_compare(s, "endscope")) return K::Endscope;
    if (case_compare(s, "include")) return K::Include;
    if (case_compare(s, "include_ffi")) return K::IncludeFfi;
    if (case_compare(s, "export")) return K::Export;
    if (case_compare(s, "print")) return K::Print;

    return K::Bogus;
}

static Token::Kind token_kind_from_type(const std::string_view s) {
    using K = Token::Kind;

    if (s == "integer") return K::TInteger;
    if (s == "boolean") return K::TBoolean;
    if (s == "real") return K::TReal;
    if (s == "char") return K::TChar;
    if (s == "string") return K::TString;
    if (s == "array") return K::TArray;

    return K::Bogus;
}

std::string Pos::to_string() const {
    return std::format("{} {} {}", row, col, span);
}

Token::Token() : kind(Kind::Bogus) {
}

Token::Token(Kind k, Pos p) : pos(p), kind(k) {
}

Token::Token(Kind k, Pos p, std::string_view s) : data(s), pos(p), kind(k) {
}

std::string Token::kind_to_string(const Kind& k) {
    switch (k) {
        case Kind::Bogus: return "!!! bogus amogus token !!!";
        case Kind::Declare: return "Declare";
        case Kind::Constant: return "Constant";
        case Kind::Output: return "Output";
        case Kind::Input: return "Input";
        case Kind::And: return "And";
        case Kind::Or: return "Or";
        case Kind::Not: return "Not";
        case Kind::If: return "If";
        case Kind::Then: return "Then";
        case Kind::Else: return "Else";
        case Kind::Endif: return "Endif";
        case Kind::Case: return "Case";
        case Kind::Of: return "Of";
        case Kind::Otherwise: return "Otherwise";
        case Kind::Endcase: return "Endcase";
        case Kind::While: return "While";
        case Kind::Do: return "Do";
        case Kind::Endwhile: return "Endwhile";
        case Kind::Repeat: return "Repeat";
        case Kind::Until: return "Until";
        case Kind::For: return "For";
        case Kind::To: return "To";
        case Kind::Step: return "Step";
        case Kind::Next: return "Next";
        case Kind::Procedure: return "Procedure";
        case Kind::Endprocedure: return "Endprocedure";
        case Kind::Call: return "Call";
        case Kind::Function: return "Function";
        case Kind::Return: return "Return";
        case Kind::Returns: return "Returns";
        case Kind::Endfunction: return "Endfunction";
        case Kind::Openfile: return "Openfile";
        case Kind::Readfile: return "Readfile";
        case Kind::Writefile: return "Writefile";
        case Kind::Closefile: return "Closefile";
        case Kind::Read: return "Read";
        case Kind::Write: return "Write";
        case Kind::Append: return "Append";
        case Kind::Include: return "Include";
        case Kind::IncludeFfi: return "IncludeFfi";
        case Kind::Export: return "Export";
        case Kind::Scope: return "Scope";
        case Kind::Endscope: return "Endscope";
        case Kind::Print: return "Print";
        case Kind::Trace: return "Trace";
        case Kind::Endtrace: return "Endtrace";
        case Kind::Assign: return "Assign";
        case Kind::Equal: return "Equal";
        case Kind::LessThan: return "LessThan";
        case Kind::GreaterThan: return "GreaterThan";
        case Kind::LessThanOrEqual: return "LessThanOrEqual";
        case Kind::GreaterThanOrEqual: return "GreaterThanOrEqual";
        case Kind::NotEqual: return "NotEqual";
        case Kind::Mul: return "Mul";
        case Kind::Div: return "Div";
        case Kind::Add: return "Add";
        case Kind::Sub: return "Sub";
        case Kind::Pow: return "Pow";
        case Kind::LeftParen: return "LeftParen";
        case Kind::RightParen: return "RightParen";
        case Kind::LeftBracket: return "LeftBracket";
        case Kind::RightBracket: return "RightBracket";
        case Kind::LeftCurly: return "LeftCurly";
        case Kind::RightCurly: return "RightCurly";
        case Kind::Colon: return "Colon";
        case Kind::Comma: return "Comma";
        case Kind::Dot: return "Dot";
        case Kind::Newline: return "Newline";
        case Kind::LiteralString: return "LiteralString";
        case Kind::LiteralChar: return "LiteralChar";
        case Kind::LiteralNumber: return "LiteralNumber";
        case Kind::True: return "True";
        case Kind::False: return "False";
        case Kind::Null: return "Null";
        case Kind::Ident: return "Ident";
        case Kind::Comment: return "Comment";
        case Kind::TInteger: return "TInteger";
        case Kind::TBoolean: return "TBoolean";
        case Kind::TReal: return "TReal";
        case Kind::TChar: return "TChar";
        case Kind::TString: return "TString";
        case Kind::TArray: return "TArray";
    };
}

std::string Token::to_string() const {
    return std::format("token({})", kind_to_string(kind));
}

// FILE* f = stdout
void Token::print(FILE* f) const {
    std::string inner;

    switch (kind) {
        case Kind::LiteralString: {
            inner = std::format("\"{}\"", data);
        } break;
        case Kind::LiteralChar: {
            inner = std::format("\'{}\'", data);
        } break;
        case Kind::LiteralNumber: {
            inner = data;
        } break;
        case Kind::Ident: {
            inner = std::format("I\"{}\"", data);
        } break;
        default: {
            inner = std::format("<{}>", kind_to_string(kind));
        } break;
    }

    std::println(f, "token[{}]: {}", pos.to_string(), inner);
}

Lexer::Lexer(const std::string& src) : src(src), row(0), col(0) {
    reset();
}

inline char32_t Lexer::get_cur_cp() const {
    char32_t res;
    assert(res = utf8::decode(&CUR) >= 0);
    return res;
}

inline bool Lexer::in_bounds() const {
    return cur < src.length();
}

Pos Lexer::pos(u16 span) const {
    return Pos{row, static_cast<u16>(col - span), span};
}

Pos Lexer::pos_here(u16 span) const {
    return Pos{row, static_cast<u16>(col), span};
}

inline bool Lexer::is_separator(char ch) const {
    return strchr("[]{}();:,", ch) != NULL;
}

inline bool Lexer::is_operator_start(const char* ch) const {
    if (strchr("%+-*/<>=^", *ch) != NULL) return true;

    if (cur + 2 >= src.length()) return false;

    int32_t res = utf8::decode(ch);
    assert(res >= 0);

    // unicode ←
    switch (res) {
        case 0x2190:
        case 0xf0ac: return true;
        default: return false;
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
    if (!in_bounds()) return;

    while (in_bounds() && isspace(CUR) && CUR != '\n')
        cur++;

    trim_comments();
}

void Lexer::trim_comments() {
    std::string_view pair;

    if (cur + 2 > src.length()) return;

    pair = src.substr(cur, 2);
    if (pair == "/*") {
        cur += 2;

        while (in_bounds() && src.substr(cur, 2) != "*/") {
            if (CUR == '\n')
                bump_newline();
            else
                cur++;
        }

        // found the delimiter
        cur++;
        trim_spaces();
    } else if (pair == "#!" || pair == "//") {
        cur += 2;
        while (in_bounds() && CUR != '\n')
            cur++;
        // don't skip the newline, the whitespace function will do it
        trim_spaces();
    }
}

static bool is_number(const std::string_view word) {
    bool found_decimal = false;

    for (const char& c : word) {
        if (isdigit(c)) continue;

        if (c == '.') {
            if (found_decimal)
                return false;
            else
                found_decimal = true;

            continue;
        }

        return false;
    }

    if (found_decimal && word.length() == 1) return false;

    return true;
}

static bool is_ident(const std::string_view word) {
    if (!isalpha(word[0]) && word[0] == '_') {
        return false;
    }

    for (const char c : word) {
        if (!isalnum(c) && c != '_' && c != '.') return false;
    }

    return true;
}

std::string_view Lexer::next_word() {
    const char* DELIMS = "\"'";
    u64 begin = cur, len = 0;
    char cur_ch, delim = src[begin];

    bool stop = false, is_delimited = strchr(DELIMS, delim);

    if (is_delimited) {
        len++;
        cur++;
    }

    do {
        stop = false;
        if (!in_bounds()) break;

        cur_ch = CUR;
        // XXX: trust me bro it works
        if (is_delimited)
            stop = (cur_ch == delim || cur_ch == '\n');
        else
            stop =
                (is_operator_start(&CUR) || is_separator(cur_ch) || isspace(cur_ch) || strchr(DELIMS, cur_ch) != NULL);

        if (cur_ch == '\\') {
            len++;
            cur++;
        }

        if (stop) break;

        len++;
        cur++;
    } while (true);

    if (is_delimited) {
        if (!in_bounds() || isspace(CUR))
            throw BCError(BCError::Kind::Syntax, pos(len), "could not find ending delimiter in literal");

        len++;
        cur++;
    }

    return src.substr(begin, len);
}

auto Lexer::next_multi_symbol() -> std::optional<Token> {
    if (!is_operator_start(&CUR)) return {};

    std::string pair;
    if (cur + 2 < src.length())
        pair = src.substr(cur, 3);
    else if (cur + 1 < src.length())
        pair = src.substr(cur, 2);
    else
        return {};

    auto k = token_kind_from_multi_op(pair);
    if (k != Token::Kind::Bogus) {
        cur += pair.length();
        return Token(k, pos(pair.length()));
    }

    return {};
}

auto Lexer::next_single_symbol() -> std::optional<Token> {
    if (!is_operator_start(&CUR)) return {};

    auto k = token_kind_from_single_op(CUR);
    if (k != Token::Kind::Bogus) {
        cur++;
        return Token(k, pos(1));
    }

    return {};
}

auto Lexer::next_keyword(const std::string_view word) -> std::optional<Token> {
    static const std::unordered_map<std::string, Token::Kind> KEYWORDS = {};

    if (util::case_consistent(word)) {
        auto k = token_kind_from_keyword(word);
        if (k != Token::Kind::Bogus) return Token(k, pos(word.length()));
    }

    return {};
}

auto Lexer::next_type(const std::string_view word) -> std::optional<Token> {
    static const std::unordered_map<std::string, Token::Kind> TYPES = {};

    if (util::case_consistent(word)) {
        auto k = token_kind_from_type(word);
        auto p = pos(word.length());
        if (k != Token::Kind::Bogus) return Token(k, p);

        if (util::case_compare(word, "endfor")) {
            throw BCError(BCError::Kind::Syntax, p, "ENDFOR is not a valid keyword!",
                          "Please use NEXT <your counter> to end a FOR loop instead.");
        }
    }

    return {};
}

auto Lexer::next_literal(const std::string_view word) -> std::optional<Token> {
    using K = Token::Kind;

    if (word[0] == '"' || word[0] == '\"') {
        if (word.length() == 1) std::unreachable();

        auto res = word.substr(1, word.length() - 2);
        auto k = word[0] == '"' ? Token::Kind::LiteralString : Token::Kind::LiteralChar;

        return Token(k, pos(word.length()), res);
    }

    if (is_number(word)) {
        return Token(K::LiteralNumber, pos(word.length()), word);
    } else if (isdigit(word[0])) {
        throw BCError(BCError::Kind::Syntax, pos(word.length()), "invalid number literal");
    }

    if (util::case_consistent(word)) {
        if (util::case_compare(word, "true")) {
            return Token(K::True, pos(word.length()));
        } else if (util::case_compare(word, "false")) {
            return Token(K::False, pos(word.length()));
        } else if (util::case_compare(word, "null")) {
            return Token(K::Null, pos(word.length()));
        }
    }

    return {};
}

auto Lexer::next_ident(const std::string_view word) -> std::optional<Token> {
    auto p = pos(word.length());

    if (is_ident(word)) {
        return Token(Token::Kind::Ident, p, word);
    } else {
        throw BCError(BCError::Kind::Syntax, p, "invalid identifier or symbol");
    }
}

auto Lexer::next_token() -> std::optional<Token> {
    using K = Token::Kind;

    trim_spaces();

    if (!in_bounds()) return {};

    if (CUR == '\n') {
        auto t = Token(K::Newline, pos_here(1));
        bump_newline();
        return t;
    }

    std::optional<Token> res;

    if ((res = next_multi_symbol())) return res;

    if ((res = next_single_symbol())) return res;

    std::string_view word = next_word();

    if ((res = next_keyword(word))) return res;

    if ((res = next_literal(word))) return res;

    if ((res = next_type(word))) return res;

    return next_ident(word);
}

auto Lexer::tokenize() -> std::vector<Token> {
    std::vector<Token> res{};

    while (in_bounds()) {
        auto t = next_token();
        if (!t) break;
        res.push_back(t.value());
    }

    res.push_back(Token(Token::Kind::Newline, pos_here(1)));
    return res;
}

}; // namespace beancode::lexer
