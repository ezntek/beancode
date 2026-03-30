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
#include "error.h"
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <string.h>

#include "a_string_slice.h"
#include "a_vector.h"
#include "lexer.h"
#include "lexer_types.h"

AV_DECL(BCToken, Tokens);

#define CUR       (l->src[l->cur])
#define IN_BOUNDS (l->cur < l->src_len)
#define POS(sp)                                                                \
    (BCPos) {                                                                  \
        .row = l->row, .col = l->cur - l->bol + 1 - (sp), .span = (sp)         \
    }

#define POS_HERE(sp)                                                           \
    (BC; Pos) {                                                                \
        .row = l->row, .col = l->cur - l->bol + 1, .span = (sp)                \
    }

#define BUMP_NEWLINE                                                           \
    do {                                                                       \
        l->bol = ++l->cur;                                                     \
        l->row++;                                                              \
    } while (0)

static BCTokenKind token_kind_from_single_op(char ch);
static BCTokenKind token_kind_from_multi_op(const a_string_slice s);
static BCTokenKind token_kind_from_keyword(const a_string_slice s);

static void trim_spaces(BCLexer* l);
static void trim_comments(BCLexer* l);
static bool is_number(BCLexer* l, a_string_slice word);
static bool is_ident(BCLexer* l, a_string_slice word);
static bool is_operator_start(BCLexer* l, const char* start);
static bool is_separator(char ch);

static bool next_word(BCLexer* l, a_string_slice* out);
static bool next_multi_symbol(BCLexer* l);
static bool next_single_symbol(BCLexer* l);
static bool next_keyword(BCLexer* l);
static bool next_literal(BCLexer* l);
static bool next_ident(BCLexer* l);

static BCTokenKind token_kind_from_single_op(char ch) {
    switch (ch) {
        case '{': return BC_TOKEN_LCURLY;
        case '}': return BC_TOKEN_RCURLY;
        case '[': return BC_TOKEN_LBRACKET;
        case ']': return BC_TOKEN_RBRACKET;
        case '(': return BC_TOKEN_LPAREN;
        case ')': return BC_TOKEN_RPAREN;
        case ':': return BC_TOKEN_COLON;
        case ';': return BC_TOKEN_NEWLINE;
        case ',': return BC_TOKEN_COMMA;
        case '=': return BC_TOKEN_EQ;
        case '<': return BC_TOKEN_LT;
        case '>': return BC_TOKEN_GT;
        case '*': return BC_TOKEN_MUL;
        case '/': return BC_TOKEN_DIV;
        case '+': return BC_TOKEN_ADD;
        case '-': return BC_TOKEN_SUB;
        case '^': return BC_TOKEN_POW;
        default: return BC_TOKEN_BOGUS;
    }
}

static BCTokenKind token_kind_from_multi_op(const a_string_slice s) {
    // U+2190 ←
    if (!strncmp(s.data, "\xe2\x86\x90", s.len)) return BC_TOKEN_ASSIGN;
    // U+F0AC
    if (!strncmp(s.data, "\xef\x82\xac", s.len)) return BC_TOKEN_ASSIGN;
    if (!strncmp(s.data, "<-", s.len)) return BC_TOKEN_ASSIGN;
    if (!strncmp(s.data, "<>", s.len)) return BC_TOKEN_NEQ;
    if (!strncmp(s.data, ">=", s.len)) return BC_TOKEN_GEQ;
    if (!strncmp(s.data, "<=", s.len)) return BC_TOKEN_LEQ;

    return BC_TOKEN_BOGUS;
}

static BCTokenKind token_kind_from_keyword(const a_string_slice s) {
    if (!strncmp(s.data, "declare", s.len)) return BC_TOKEN_DECLARE;
    if (!strncmp(s.data, "constant", s.len)) return BC_TOKEN_CONSTANT;
    if (!strncmp(s.data, "output", s.len)) return BC_TOKEN_OUTPUT;
    if (!strncmp(s.data, "input", s.len)) return BC_TOKEN_INPUT;
    if (!strncmp(s.data, "and", s.len)) return BC_TOKEN_AND;
    if (!strncmp(s.data, "or", s.len)) return BC_TOKEN_OR;
    if (!strncmp(s.data, "not", s.len)) return BC_TOKEN_NOT;
    if (!strncmp(s.data, "if", s.len)) return BC_TOKEN_IF;
    if (!strncmp(s.data, "then", s.len)) return BC_TOKEN_THEN;
    if (!strncmp(s.data, "else", s.len)) return BC_TOKEN_ELSE;
    if (!strncmp(s.data, "endif", s.len)) return BC_TOKEN_ENDIF;
    if (!strncmp(s.data, "case", s.len)) return BC_TOKEN_CASE;
    if (!strncmp(s.data, "of", s.len)) return BC_TOKEN_OF;
    if (!strncmp(s.data, "otherwise", s.len)) return BC_TOKEN_OTHERWISE;
    if (!strncmp(s.data, "endcase", s.len)) return BC_TOKEN_ENDCASE;
    if (!strncmp(s.data, "while", s.len)) return BC_TOKEN_WHILE;
    if (!strncmp(s.data, "do", s.len)) return BC_TOKEN_DO;
    if (!strncmp(s.data, "endwhile", s.len)) return BC_TOKEN_ENDWHILE;
    if (!strncmp(s.data, "repeat", s.len)) return BC_TOKEN_REPEAT;
    if (!strncmp(s.data, "until", s.len)) return BC_TOKEN_UNTIL;
    if (!strncmp(s.data, "for", s.len)) return BC_TOKEN_FOR;
    if (!strncmp(s.data, "to", s.len)) return BC_TOKEN_TO;
    if (!strncmp(s.data, "step", s.len)) return BC_TOKEN_STEP;
    if (!strncmp(s.data, "next", s.len)) return BC_TOKEN_NEXT;
    if (!strncmp(s.data, "procedure", s.len)) return BC_TOKEN_PROCEDURE;
    if (!strncmp(s.data, "endprocedure", s.len)) return BC_TOKEN_ENDPROCEDURE;
    if (!strncmp(s.data, "call", s.len)) return BC_TOKEN_CALL;
    if (!strncmp(s.data, "function", s.len)) return BC_TOKEN_FUNCTION;
    if (!strncmp(s.data, "returns", s.len)) return BC_TOKEN_RETURNS;
    if (!strncmp(s.data, "return", s.len)) return BC_TOKEN_RETURN;
    if (!strncmp(s.data, "endfunction", s.len)) return BC_TOKEN_ENDFUNCTION;
    if (!strncmp(s.data, "openfile", s.len)) return BC_TOKEN_OPENFILE;
    if (!strncmp(s.data, "readfile", s.len)) return BC_TOKEN_READFILE;
    if (!strncmp(s.data, "writefile", s.len)) return BC_TOKEN_WRITEFILE;
    if (!strncmp(s.data, "closefile", s.len)) return BC_TOKEN_CLOSEFILE;
    if (!strncmp(s.data, "read", s.len)) return BC_TOKEN_READ;
    if (!strncmp(s.data, "write", s.len)) return BC_TOKEN_WRITE;
    if (!strncmp(s.data, "append", s.len)) return BC_TOKEN_APPEND;
    if (!strncmp(s.data, "trace", s.len)) return BC_TOKEN_TRACE;
    if (!strncmp(s.data, "endtrace", s.len)) return BC_TOKEN_ENDTRACE;
    if (!strncmp(s.data, "scope", s.len)) return BC_TOKEN_SCOPE;
    if (!strncmp(s.data, "endscope", s.len)) return BC_TOKEN_ENDSCOPE;
    if (!strncmp(s.data, "include", s.len)) return BC_TOKEN_INCLUDE;
    if (!strncmp(s.data, "export", s.len)) return BC_TOKEN_EXPORT;
    if (!strncmp(s.data, "print", s.len)) return BC_TOKEN_PRINT;

    return BC_TOKEN_BOGUS;
}

static void trim_spaces(BCLexer* l) {
    if (!IN_BOUNDS) return;

    while (IN_BOUNDS && isspace(CUR) && CUR != '\n')
        l->cur++;

    trim_comments(l);
}

static void trim_comments(BCLexer* l) {
    if (l->cur + 2 > l->src_len) return;

    if (!strncmp(&CUR, "/*", 2)) {
        l->cur += 2;

        while (IN_BOUNDS && strncmp(&CUR, "/*", 2)) {
            if (CUR == '\n')
                BUMP_NEWLINE;
            else
                l->cur++;
        }

        l->cur++;
        trim_spaces(l);
    } else if (!strncmp(&CUR, "//", 2) || !strncmp(&CUR, "#!", 2)) {
        l->cur += 2;

        while (IN_BOUNDS && CUR != '\n')
            l->cur++;

        trim_spaces(l);
    }
}

static bool is_number(BCLexer* l, a_string_slice word) {
    bool found_decimal = false;
    char cur;

    for (usize i = 0; i < word.len; i++) {
        cur = word.data[i];
        if (isdigit(cur)) continue;

        if (cur == '.') {
            if (found_decimal)
                return false;
            else
                found_decimal = true;

            continue;
        }

        return false;
    }

    if (found_decimal && word.len == 1) return false;

    return true;
}

static bool is_ident(BCLexer* l, a_string_slice word) {
    if (!isalpha(*word.data) && *word.data == '_') return false;

    char cur;
    for (usize i = 0; i < word.len; i++) {
        cur = word.data[i];
        if (!isalnum(cur) && cur != '_' && cur != '.') return false;
    }

    return true;
}

static bool is_operator_start(BCLexer* l, const char* start) {
    // NOTE: we catch % for error
    if (strchr("%+-*/<>=^", *start) != NULL) return true;

    if (l->cur + 2 >= l->src_len) return false;

    // U+2190 ← , U+F0AC
    if (!strcmp(&CUR, "\xe2\x86\x90") || !strcmp(&CUR, "\xef\x82\xac"))
        return true;

    return false;
}

static bool is_separator(char ch) {
    return strchr("[]{}();:,", ch) != NULL;
}

static bool next_word(BCLexer* l, a_string_slice* out) {
    static const char* DELIMS = "\"'";

    usize begin = l->cur, len = 0;
    char cur, delim = l->src[begin];
    bool stop = false, is_delimited = strchr(DELIMS, delim);

    if (is_delimited) {
        len++;
        l->cur++;
    }

    do {
        if (!IN_BOUNDS) break;

        cur = CUR;
        if (is_delimited)
            stop = (cur == delim || cur == '\n');
        else
            stop = (is_operator_start(l, &CUR) || is_separator(cur) ||
                    isspace(cur) || strchr(DELIMS, cur));

        if (cur == '\\') {
            len++;
            cur++;
        }

        if (stop) break;

        len++;
        cur++;
    } while (true);

    if (is_delimited) {
        if (!IN_BOUNDS || isspace(cur)) {
            l->error =
                bc_error_new_cstr(BC_ERROR_SYNTAX, POS(len),
                                  "could not find ending delimiter in literal");
            return false;
        }

        len++;
        l->cur++;
    }

    out->len = len;
    out->data = &l->src[begin];

    return true;
}

static bool next_multi_symbol(BCLexer* l) {
    if (!is_operator_start(l, &CUR)) return false;

    a_string_slice chunk = {.data = &CUR};
    for (int s = 3; s >= 2; s--) {
        if (l->cur + (s - 1) < l->src_len) chunk.len = s;

        BCTokenKind k = token_kind_from_multi_op(chunk);
        if (k != BC_TOKEN_BOGUS) {
            l->cur += s;
            l->token = (BCToken){
                .kind = k,
                .pos = POS(s),
            };
            return true;
        }
    }

    return false;
}

static bool next_single_symbol(BCLexer* l) {
    if (!is_operator_start(l, &CUR) && !is_separator(CUR)) return false;

    BCTokenKind k = token_kind_from_single_op(CUR);

    if (k != BC_TOKEN_BOGUS) {
        l->cur++;
        l->token = (BCToken){
            .kind = k,
            .pos = POS(1),
        };
        return true;
    }

    return false;
}

static bool next_keyword(BCLexer* l);

static bool next_literal(BCLexer* l);

static bool next_ident(BCLexer* l);

// public API

BCLexer bc_lexer_new(a_string_slice src) {
    BCLexer l = {.src = src.data, .src_len = src.len};

    bc_lexer_reset(&l);

    return l;
}

void bc_lexer_reset(BCLexer* l) {
    l->row = 1;
    l->cur = 0;
    l->bol = 0;
    l->token = (BCToken){0};
    l->error = (BCError){0};
}

bool bc_lexer_next_token(BCLexer* l) {
    trim_spaces(l);

    return false;
}

usize bc_lexer_tokenize(BCLexer* l, BCToken** out);
