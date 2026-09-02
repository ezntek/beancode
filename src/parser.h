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
#ifndef BC_PARSER_H
#define BC_PARSER_H

#include "lexer_types.h"

typedef enum {
    BC_PARSER_MODE_BEANCODE = 0,
    BC_PARSER_MODE_BEANBEAN = 1,
    BC_PARSER_MODE_COOKEDBEAN = 2,
} BCParserMode;

typedef struct {
    // should be alive the entire time the parser is
    BCTokenArray *tokens;
    u32 cur;
    // BCParserMode
    u16 mode;
    bool preserve_trivia;
    BCASTStorage storage;
} BCParser;

BCParser bc_parser_new(BCTokenArray *tokens, BCParserMode mode,
                       bool preserve_trivia);

#endif // BC_PARSER_H
