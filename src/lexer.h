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
#ifndef BC_LEXER_H
#define BC_LEXER_H

#include "a_string_slice.h"
#include "common.h"
#include "error.h"

typedef struct {
    const char* src;
    usize src_len;
    usize cur;
    usize bol;
    u32 row;

    // current token being worked on
    BCToken token;

    // current error
    BCError error;
} BCLexer;

BCLexer bc_lexer_new(a_string_slice src);

void bc_lexer_reset(BCLexer* l);

// returns false on error and sets l->error to a valid value. returns true on
// success and sets l->token to a valid value.
bool bc_lexer_next_token(BCLexer* l);

// returns length of out buf
usize bc_lexer_tokenize(BCLexer* l, BCToken** out);

#endif // BC_LEXER_H
