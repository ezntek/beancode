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
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <string.h>
#include <strings.h>

#include "a_string_slice.h"
#include "a_vector.h"
#include "error.h"
#include "lexer.h"
#include "lexer_types.h"

AV_DECL(BCToken, Tokens)

#define CUR (l->src[l->cur])
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

static void trim_spaces(BCLexer *l);
static void trim_comments(BCLexer *l);
static bool is_number(a_string_slice word);
static bool is_ident(a_string_slice word);
static bool is_operator_start(BCLexer *l, const char *start);
static bool is_separator(char ch);

static bool next_word(BCLexer *l, a_string_slice *out);
static bool next_multi_symbol(BCLexer *l);
static bool next_single_symbol(BCLexer *l);
static bool next_keyword(BCLexer *l, a_string_slice word);
static bool next_literal(BCLexer *l, a_string_slice word);
static bool next_ident(BCLexer *l, a_string_slice word);

static BCTokenKind token_kind_from_single_op(char ch) {
    switch (ch) {
    case '{':
        return BC_TOKEN_LCURLY;
    case '}':
        return BC_TOKEN_RCURLY;
    case '[':
        return BC_TOKEN_LBRACKET;
    case ']':
        return BC_TOKEN_RBRACKET;
    case '(':
        return BC_TOKEN_LPAREN;
    case ')':
        return BC_TOKEN_RPAREN;
    case ':':
        return BC_TOKEN_COLON;
    case ';':
        return BC_TOKEN_NEWLINE;
    case ',':
        return BC_TOKEN_COMMA;
    case '=':
        return BC_TOKEN_EQ;
    case '<':
        return BC_TOKEN_LT;
    case '>':
        return BC_TOKEN_GT;
    case '*':
        return BC_TOKEN_MUL;
    case '/':
        return BC_TOKEN_DIV;
    case '+':
        return BC_TOKEN_ADD;
    case '-':
        return BC_TOKEN_SUB;
    case '^':
        return BC_TOKEN_POW;
    default:
        return BC_TOKEN_BOGUS;
    }
}

static BCTokenKind token_kind_from_multi_op(const a_string_slice s) {
    // U+2190 ←
    if (!strncmp(s.data, "\xe2\x86\x90", s.len))
        return BC_TOKEN_ASSIGN;
    // U+F0AC
    if (!strncmp(s.data, "\xef\x82\xac", s.len))
        return BC_TOKEN_ASSIGN;
    if (!strncmp(s.data, "<-", s.len))
        return BC_TOKEN_ASSIGN;
    if (!strncmp(s.data, "<>", s.len))
        return BC_TOKEN_NEQ;
    if (!strncmp(s.data, ">=", s.len))
        return BC_TOKEN_GEQ;
    if (!strncmp(s.data, "<=", s.len))
        return BC_TOKEN_LEQ;

    return BC_TOKEN_BOGUS;
}

static BCTokenKind token_kind_from_keyword(const a_string_slice s) {
    if (ass_equal_nocase_cstr(s, "declare"))
        return BC_TOKEN_DECLARE;
    if (ass_equal_nocase_cstr(s, "constant"))
        return BC_TOKEN_CONSTANT;
    if (ass_equal_nocase_cstr(s, "output"))
        return BC_TOKEN_OUTPUT;
    if (ass_equal_nocase_cstr(s, "input"))
        return BC_TOKEN_INPUT;
    if (ass_equal_nocase_cstr(s, "and"))
        return BC_TOKEN_AND;
    if (ass_equal_nocase_cstr(s, "or"))
        return BC_TOKEN_OR;
    if (ass_equal_nocase_cstr(s, "not"))
        return BC_TOKEN_NOT;
    if (ass_equal_nocase_cstr(s, "if"))
        return BC_TOKEN_IF;
    if (ass_equal_nocase_cstr(s, "then"))
        return BC_TOKEN_THEN;
    if (ass_equal_nocase_cstr(s, "else"))
        return BC_TOKEN_ELSE;
    if (ass_equal_nocase_cstr(s, "endif"))
        return BC_TOKEN_ENDIF;
    if (ass_equal_nocase_cstr(s, "case"))
        return BC_TOKEN_CASE;
    if (ass_equal_nocase_cstr(s, "of"))
        return BC_TOKEN_OF;
    if (ass_equal_nocase_cstr(s, "otherwise"))
        return BC_TOKEN_OTHERWISE;
    if (ass_equal_nocase_cstr(s, "endcase"))
        return BC_TOKEN_ENDCASE;
    if (ass_equal_nocase_cstr(s, "while"))
        return BC_TOKEN_WHILE;
    if (ass_equal_nocase_cstr(s, "do"))
        return BC_TOKEN_DO;
    if (ass_equal_nocase_cstr(s, "endwhile"))
        return BC_TOKEN_ENDWHILE;
    if (ass_equal_nocase_cstr(s, "repeat"))
        return BC_TOKEN_REPEAT;
    if (ass_equal_nocase_cstr(s, "until"))
        return BC_TOKEN_UNTIL;
    if (ass_equal_nocase_cstr(s, "for"))
        return BC_TOKEN_FOR;
    if (ass_equal_nocase_cstr(s, "to"))
        return BC_TOKEN_TO;
    if (ass_equal_nocase_cstr(s, "step"))
        return BC_TOKEN_STEP;
    if (ass_equal_nocase_cstr(s, "next"))
        return BC_TOKEN_NEXT;
    if (ass_equal_nocase_cstr(s, "procedure"))
        return BC_TOKEN_PROCEDURE;
    if (ass_equal_nocase_cstr(s, "endprocedure"))
        return BC_TOKEN_ENDPROCEDURE;
    if (ass_equal_nocase_cstr(s, "call"))
        return BC_TOKEN_CALL;
    if (ass_equal_nocase_cstr(s, "function"))
        return BC_TOKEN_FUNCTION;
    if (ass_equal_nocase_cstr(s, "returns"))
        return BC_TOKEN_RETURNS;
    if (ass_equal_nocase_cstr(s, "return"))
        return BC_TOKEN_RETURN;
    if (ass_equal_nocase_cstr(s, "endfunction"))
        return BC_TOKEN_ENDFUNCTION;
    if (ass_equal_nocase_cstr(s, "openfile"))
        return BC_TOKEN_OPENFILE;
    if (ass_equal_nocase_cstr(s, "readfile"))
        return BC_TOKEN_READFILE;
    if (ass_equal_nocase_cstr(s, "writefile"))
        return BC_TOKEN_WRITEFILE;
    if (ass_equal_nocase_cstr(s, "closefile"))
        return BC_TOKEN_CLOSEFILE;
    if (ass_equal_nocase_cstr(s, "read"))
        return BC_TOKEN_READ;
    if (ass_equal_nocase_cstr(s, "write"))
        return BC_TOKEN_WRITE;
    if (ass_equal_nocase_cstr(s, "append"))
        return BC_TOKEN_APPEND;
    if (ass_equal_nocase_cstr(s, "trace"))
        return BC_TOKEN_TRACE;
    if (ass_equal_nocase_cstr(s, "endtrace"))
        return BC_TOKEN_ENDTRACE;
    if (ass_equal_nocase_cstr(s, "scope"))
        return BC_TOKEN_SCOPE;
    if (ass_equal_nocase_cstr(s, "endscope"))
        return BC_TOKEN_ENDSCOPE;
    if (ass_equal_nocase_cstr(s, "include"))
        return BC_TOKEN_INCLUDE;
    if (ass_equal_nocase_cstr(s, "export"))
        return BC_TOKEN_EXPORT;
    if (ass_equal_nocase_cstr(s, "print"))
        return BC_TOKEN_PRINT;

    return BC_TOKEN_BOGUS;
}

static void trim_spaces(BCLexer *l) {
    if (!IN_BOUNDS)
        return;

    while (IN_BOUNDS && isspace(CUR) && CUR != '\n')
        l->cur++;

    trim_comments(l);
}

static void trim_comments(BCLexer *l) {
    if (l->cur + 2 > l->src_len)
        return;

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

static bool is_number(a_string_slice word) {
    bool found_decimal = false;
    char cur;

    for (usize i = 0; i < word.len; i++) {
        cur = word.data[i];
        if (isdigit(cur))
            continue;

        if (cur == '.') {
            if (found_decimal)
                return false;

            found_decimal = true;
            continue;
        }

        return false;
    }

    if (found_decimal && word.len == 1)
        return false;

    return true;
}

static bool is_ident(a_string_slice word) {
    if (!isalpha(*word.data) && *word.data == '_')
        return false;

    char cur;
    for (usize i = 0; i < word.len; i++) {
        cur = word.data[i];
        if (!isalnum(cur) && cur != '_' && cur != '.')
            return false;
    }

    return true;
}

static bool is_operator_start(BCLexer *l, const char *start) {
    // NOTE: we catch % for error
    if (strchr("%+-*/<>=^", *start) != NULL)
        return true;

    if (l->cur + 2 >= l->src_len)
        return false;

    // U+2190 ← , U+F0AC
    if (!strcmp(&CUR, "\xe2\x86\x90") || !strcmp(&CUR, "\xef\x82\xac"))
        return true;

    return false;
}

static bool is_separator(char ch) {
    return strchr("[]{}();:,", ch) != NULL;
}

static bool next_word(BCLexer *l, a_string_slice *out) {
    static const char *DELIMS = "\"'";

    usize begin = l->cur, len = 0;
    char cur, delim = l->src[begin];
    bool stop = false, is_delimited = strchr(DELIMS, delim);

    if (is_delimited) {
        len++;
        l->cur++;
    }

    do {
        if (!IN_BOUNDS)
            break;

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

        if (stop)
            break;

        len++;
        l->cur++;
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

static bool next_multi_symbol(BCLexer *l) {
    if (!is_operator_start(l, &CUR))
        return false;

    a_string_slice chunk = {.data = &CUR};
    for (int s = 3; s >= 2; s--) {
        if (l->cur + (s - 1) < l->src_len)
            chunk.len = s;

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

static bool next_single_symbol(BCLexer *l) {
    if (!is_operator_start(l, &CUR) && !is_separator(CUR))
        return false;

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

static bool next_keyword(BCLexer *l, a_string_slice word) {
    if (!ass_case_consistent(word))
        return false;

    BCPos p = POS(word.len);
    BCTokenKind k = token_kind_from_keyword(word);
    if (k != BC_TOKEN_BOGUS) {
        l->token = (BCToken){
            .kind = k,
            .pos = p,
        };
        return true;
    }

    if (ass_equal_nocase_cstr(word, "endfor")) {
        l->error =
            bc_error_new_cstr(BC_ERROR_SYNTAX, p,
                              "ENDFOR is not a valid keyword!\nPlease use NEXT "
                              "<your counter> to end a FOR loop instead.");
        return false;
    }

    return false;
}

static bool next_literal(BCLexer *l, a_string_slice word) {
    if (ass_first(word) == '"' || ass_first(word) == '\'') {
        if (word.len == 1)
            panic("unreachable code");

        BCTokenKind k =
            ass_first(word) == '"' ? BC_TOKEN_LIT_STRING : BC_TOKEN_LIT_CHAR;

        l->token = (BCToken){
            .src_index = (u32)l->cur - word.len,
            .kind = k,
            .pos = POS(word.len),
        };
        return true;
    }

    if (is_number(word)) {
        l->token = (BCToken){
            .src_index = (u32)l->cur - word.len,
            .kind = BC_TOKEN_LIT_NUMBER,
            .pos = POS(word.len),
        };
        return true;
    } else if (isdigit(ass_first(word))) {
        l->error = bc_error_new_cstr(BC_ERROR_SYNTAX, POS(word.len),
                                     "invalid number literal");
        return false;
    }

    if (ass_case_consistent(word)) {
        l->token.pos = POS(word.len);
        if (ass_equal_nocase_cstr(word, "true")) {
            l->token.kind = BC_TOKEN_TRUE;
            return true;
        } else if (ass_equal_nocase_cstr(word, "false")) {
            l->token.kind = BC_TOKEN_FALSE;
            return true;
        } else if (ass_equal_nocase_cstr(word, "null")) {
            l->token.kind = BC_TOKEN_NULL;
            return true;
        }
    }

    return false;
}

static bool next_ident(BCLexer *l, a_string_slice word) {
    BCPos p = POS(word.len);

    if (is_ident(word)) {
        l->token = (BCToken){
            .src_index = (u32)l->cur - word.len,
            .kind = BC_TOKEN_IDENT,
            .pos = p,
        };
        return true;
    } else {
        l->error = bc_error_new_cstr(BC_ERROR_SYNTAX, p,
                                     "invalid identifier or symbol");
        return false;
    }
}

// public API

BCLexer bc_lexer_new(a_string_slice src) {
    BCLexer l = {.src = src.data, .src_len = src.len};

    bc_lexer_reset(&l);

    return l;
}

void bc_lexer_reset(BCLexer *l) {
    l->row = 1;
    l->cur = 0;
    l->bol = 0;
    l->token = (BCToken){0};
    l->error = (BCError){0};
}

BCToken *bc_lexer_next_token(BCLexer *l) {
    trim_spaces(l);

    if (!IN_BOUNDS) {
        l->token = (BCToken){
            .kind = BC_TOKEN_EOF,
            .pos = POS(1),
        };
        return &l->token;
    }

    if (CUR == '\n') {
        l->token = (BCToken){
            .kind = BC_TOKEN_NEWLINE,
            .pos = POS(1),
        };
        BUMP_NEWLINE;
        return &l->token;
    }

    if (next_multi_symbol(l))
        return &l->token;
    if (next_single_symbol(l))
        return &l->token;

    a_string_slice word = {0};
    if (!next_word(l, &word))
        return NULL;

    if (next_keyword(l, word))
        return &l->token;
    if (next_literal(l, word))
        return &l->token;
    if (next_ident(l, word))
        return &l->token;

    return NULL;
}

usize bc_lexer_tokenize(BCLexer *l, BCToken **out) {
    Tokens res = {0};
    BCToken *tok = NULL;

    bc_lexer_reset(l);

    do {
        tok = bc_lexer_next_token(l);
        if (!tok) {
            *out = NULL;
            if (res.cap)
                av_free(&res);
            return 0;
        }

        av_append(&res, *tok);
    } while (!tok || tok->kind != BC_TOKEN_EOF);

    *out = res.data;
    return res.len;
}
