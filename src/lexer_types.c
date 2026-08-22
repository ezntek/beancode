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

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "lexer_types.h"
#include "str.h"

static const char *TOKEN_KIND_TABLE[] = {
    [BC_TOKEN_BOGUS] = "!!! bogus amogus token !!!",
    [BC_TOKEN_EOF] = "eof",
    [BC_TOKEN_DECLARE] = "declare",
    [BC_TOKEN_CONSTANT] = "constant",
    [BC_TOKEN_OUTPUT] = "output",
    [BC_TOKEN_INPUT] = "input",
    [BC_TOKEN_AND] = "and",
    [BC_TOKEN_OR] = "or",
    [BC_TOKEN_NOT] = "not",
    [BC_TOKEN_IF] = "if",
    [BC_TOKEN_THEN] = "then",
    [BC_TOKEN_ELSE] = "else",
    [BC_TOKEN_ENDIF] = "endif",
    [BC_TOKEN_CASE] = "case",
    [BC_TOKEN_OF] = "of",
    [BC_TOKEN_OTHERWISE] = "otherwise",
    [BC_TOKEN_ENDCASE] = "endcase",
    [BC_TOKEN_WHILE] = "while",
    [BC_TOKEN_DO] = "do",
    [BC_TOKEN_ENDWHILE] = "endwhile",
    [BC_TOKEN_REPEAT] = "repeat",
    [BC_TOKEN_UNTIL] = "until",
    [BC_TOKEN_FOR] = "for",
    [BC_TOKEN_TO] = "to",
    [BC_TOKEN_STEP] = "step",
    [BC_TOKEN_NEXT] = "next",
    [BC_TOKEN_PROCEDURE] = "procedure",
    [BC_TOKEN_ENDPROCEDURE] = "endprocedure",
    [BC_TOKEN_CALL] = "call",
    [BC_TOKEN_FUNCTION] = "function",
    [BC_TOKEN_RETURN] = "return",
    [BC_TOKEN_RETURNS] = "returns",
    [BC_TOKEN_ENDFUNCTION] = "endfunction",
    [BC_TOKEN_OPENFILE] = "openfile",
    [BC_TOKEN_READFILE] = "readfile",
    [BC_TOKEN_WRITEFILE] = "writefile",
    [BC_TOKEN_CLOSEFILE] = "closefile",
    [BC_TOKEN_READ] = "read",
    [BC_TOKEN_WRITE] = "write",
    [BC_TOKEN_APPEND] = "append",
    [BC_TOKEN_INCLUDE] = "include",
    [BC_TOKEN_EXPORT] = "export",
    [BC_TOKEN_SCOPE] = "scope",
    [BC_TOKEN_ENDSCOPE] = "endscope",
    [BC_TOKEN_PRINT] = "print",
    [BC_TOKEN_TRACE] = "trace",
    [BC_TOKEN_ENDTRACE] = "endtrace",
    [BC_TOKEN_ASSIGN] = "assign",
    [BC_TOKEN_EQ] = "eq",
    [BC_TOKEN_LT] = "lt",
    [BC_TOKEN_GT] = "gt",
    [BC_TOKEN_LEQ] = "leq",
    [BC_TOKEN_GEQ] = "geq",
    [BC_TOKEN_NEQ] = "neq",
    [BC_TOKEN_MUL] = "mul",
    [BC_TOKEN_DIV] = "div",
    [BC_TOKEN_ADD] = "add",
    [BC_TOKEN_SUB] = "sub",
    [BC_TOKEN_POW] = "pow",
    [BC_TOKEN_LPAREN] = "lparen",
    [BC_TOKEN_RPAREN] = "rparen",
    [BC_TOKEN_LBRACKET] = "lbracket",
    [BC_TOKEN_RBRACKET] = "rbracket",
    [BC_TOKEN_LCURLY] = "lcurly",
    [BC_TOKEN_RCURLY] = "rcurly",
    [BC_TOKEN_COLON] = "colon",
    [BC_TOKEN_COMMA] = "comma",
    [BC_TOKEN_DOT] = "dot",
    [BC_TOKEN_NEWLINE] = "newline",
    [BC_TOKEN_LIT_STRING] = "lit_string",
    [BC_TOKEN_LIT_CHAR] = "lit_char",
    [BC_TOKEN_LIT_NUMBER] = "lit_number",
    [BC_TOKEN_TRUE] = "true",
    [BC_TOKEN_FALSE] = "false",
    [BC_TOKEN_NULL] = "null",
    [BC_TOKEN_IDENT] = "ident",
    [BC_TOKEN_INTEGER] = "integer",
    [BC_TOKEN_BOOLEAN] = "boolean",
    [BC_TOKEN_REAL] = "real",
    [BC_TOKEN_CHAR] = "char",
    [BC_TOKEN_STRING] = "string",
    [BC_TOKEN_ARRAY] = "array",
};

static char bc_token_full_buf[512] = {0};
static char bc_pos_buf[256] = {0};
static char bc_token_buf[256] = {0};

str_view bc_pos_to_string_slice(const BCPos *p) {
    usize len = snprintf(bc_pos_buf, sizeof(bc_pos_buf), "%u %u %u", p->row,
                         p->col, p->span);
    return (str_view){.data = bc_pos_buf, .len = len};
}

str bc_pos_to_string(const BCPos *p) {
    return str_format("[%u %u %u]", p->row, p->col, p->span);
}

str_view bc_token_kind_to_string_slice(BCTokenKind k) {
    return sv_from_cstr(TOKEN_KIND_TABLE[k]);
}

str bc_token_kind_to_string(BCTokenKind k) {
    return mstr(TOKEN_KIND_TABLE[k]);
}

str_view bc_token_to_string_slice(const BCToken *t) {
    str_view k = bc_token_kind_to_string_slice(t->kind),
             p = bc_pos_to_string_slice(&t->pos);

    usize len = snprintf(bc_token_buf, sizeof(bc_token_buf),
                         "token[%.*s]: %.*s", str_fmt(&p), str_fmt(&k));

    return (str_view){.data = bc_token_buf, .len = len};
}

str bc_token_to_string(const BCToken *t) {
    return str_from_sv(bc_token_to_string_slice(t));
}

str_view bc_token_to_string_slice_full(const BCToken *t, const str_view src) {
    str_view k = bc_token_kind_to_string_slice(t->kind),
             p = bc_pos_to_string_slice(&t->pos);
    usize len = 0;

    switch (t->kind) {
    case BC_TOKEN_LIT_CHAR:
    case BC_TOKEN_LIT_NUMBER:
    case BC_TOKEN_LIT_STRING:
    case BC_TOKEN_IDENT: {
        assert(sv_valid(&src));
        assert(t->src_index + t->pos.span <= src.len);
        len = snprintf(bc_token_full_buf, sizeof(bc_token_buf),
                       "token[%.*s]: {%.*s}", str_fmt(&p), (int)t->pos.span,
                       src.data + (usize)t->src_index);
    } break;
    default: {
        len = snprintf(bc_token_full_buf, sizeof(bc_token_buf),
                       "token[%.*s]: <%.*s>", str_fmt(&p), str_fmt(&k));
    } break;
    }

    return (str_view){.data = bc_token_full_buf, .len = len};
}

str bc_token_to_string_full(const BCToken *t, const str_view src) {
    return str_from_sv(bc_token_to_string_slice_full(t, src));
}
