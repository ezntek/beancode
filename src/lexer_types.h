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
#ifndef BC_LEXER_TYPES_H
#define BC_LEXER_TYPES_H

#include "common.h"
#include "str.h"
#include "vec.h"

typedef struct {
    u32 row;
    u16 col;
    u16 span;
} BCPos;

// on every call, the result of the previous call is destroyed as we use one
// static buffer
str_view bc_pos_to_string_slice(const BCPos *p);

str bc_pos_to_string(const BCPos *p);

typedef enum {
    BC_TOKEN_BOGUS = 0,
    BC_TOKEN_EOF,
    BC_TOKEN_DECLARE,
    BC_TOKEN_CONSTANT,
    BC_TOKEN_OUTPUT,
    BC_TOKEN_INPUT,
    BC_TOKEN_AND,
    BC_TOKEN_OR,
    BC_TOKEN_NOT,
    BC_TOKEN_IF,
    BC_TOKEN_THEN,
    BC_TOKEN_ELSE,
    BC_TOKEN_ENDIF,
    BC_TOKEN_CASE,
    BC_TOKEN_OF,
    BC_TOKEN_OTHERWISE,
    BC_TOKEN_ENDCASE,
    BC_TOKEN_WHILE,
    BC_TOKEN_DO,
    BC_TOKEN_ENDWHILE,
    BC_TOKEN_REPEAT,
    BC_TOKEN_UNTIL,
    BC_TOKEN_FOR,
    BC_TOKEN_TO,
    BC_TOKEN_STEP,
    BC_TOKEN_NEXT,
    BC_TOKEN_PROCEDURE,
    BC_TOKEN_ENDPROCEDURE,
    BC_TOKEN_CALL,
    BC_TOKEN_FUNCTION,
    BC_TOKEN_RETURN,
    BC_TOKEN_RETURNS,
    BC_TOKEN_ENDFUNCTION,
    BC_TOKEN_OPENFILE,
    BC_TOKEN_READFILE,
    BC_TOKEN_WRITEFILE,
    BC_TOKEN_CLOSEFILE,
    BC_TOKEN_READ,
    BC_TOKEN_WRITE,
    BC_TOKEN_APPEND,
    BC_TOKEN_INCLUDE,
    BC_TOKEN_EXPORT,
    BC_TOKEN_SCOPE,
    BC_TOKEN_ENDSCOPE,
    BC_TOKEN_PRINT,
    BC_TOKEN_TRACE,
    BC_TOKEN_ENDTRACE,
    BC_TOKEN_ASSIGN,
    BC_TOKEN_EQ,
    BC_TOKEN_LT,
    BC_TOKEN_GT,
    BC_TOKEN_LEQ,
    BC_TOKEN_GEQ,
    BC_TOKEN_NEQ,
    BC_TOKEN_MUL,
    BC_TOKEN_DIV,
    BC_TOKEN_ADD,
    BC_TOKEN_SUB,
    BC_TOKEN_POW,
    BC_TOKEN_LPAREN,
    BC_TOKEN_RPAREN,
    BC_TOKEN_LBRACKET,
    BC_TOKEN_RBRACKET,
    BC_TOKEN_LCURLY,
    BC_TOKEN_RCURLY,
    BC_TOKEN_COLON,
    BC_TOKEN_COMMA,
    BC_TOKEN_DOT,
    BC_TOKEN_NEWLINE,
    BC_TOKEN_LIT_STRING,
    BC_TOKEN_LIT_CHAR,
    BC_TOKEN_LIT_NUMBER,
    BC_TOKEN_TRUE,
    BC_TOKEN_FALSE,
    BC_TOKEN_NULL,
    BC_TOKEN_IDENT,
    BC_TOKEN_INTEGER,
    BC_TOKEN_BOOLEAN,
    BC_TOKEN_REAL,
    BC_TOKEN_CHAR,
    BC_TOKEN_STRING,
    BC_TOKEN_ARRAY,
} BCTokenKind;

// on every call, the result of the previous call is destroyed as we use one
// static buffer
str_view bc_token_kind_to_string_slice(BCTokenKind k);

str bc_token_kind_to_string(BCTokenKind k);

typedef struct {
    BCTokenKind kind;
    BCPos pos;
    // index into src string
    u32 src_index;
} BCToken;

VEC_DECL(BCToken, BCTokenArray);

// on every call, the result of the previous call is destroyed as we use one
// static buffer
str_view bc_token_to_string_slice(const BCToken *t);

str bc_token_to_string(const BCToken *t);

str_view bc_token_to_string_slice_full(const BCToken *t, const str_view src);

str bc_token_to_string_full(const BCToken *t, const str_view src);

#endif // BC_LEXER_TYPES_H
