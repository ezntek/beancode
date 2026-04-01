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
#ifndef BC_ERROR_H
#define BC_ERROR_H

#include <stdbool.h>

#include "a_string.h"
#include "a_string_slice.h"
#include "lexer_types.h"

typedef enum {
    BC_ERROR_EOF = 0,
    BC_ERROR_SYNTAX,
    BC_ERROR_RUNTIME,
} BCErrorKind;

a_string_slice bc_error_kind_to_string_slice(BCErrorKind k);
a_string bc_error_kind_to_string(BCErrorKind k);

typedef struct {
    BCErrorKind kind;
    BCPos pos;
    a_string msg; // rows delimited by 0xA
} BCError;

BCError bc_error_new(BCErrorKind k, BCPos p, a_string msg);
BCError bc_error_new_cstr(BCErrorKind k, BCPos p, const char* msg);
BCError bc_error_new_string_slice(BCErrorKind k, BCPos p, a_string_slice msg);

struct __bc_error_print_opts {
    a_string_slice file_name;
    a_string_slice src;
    FILE* f;
    bool no_color;
};

// void bc_error_print(BCError* err, a_string_slice file_name, ...)
#define bc_error_print(err, ...)                                               \
    __bc_error_print_impl(                                                     \
        (err), (struct __bc_error_print_opts){.file_name = __VA_ARGS__})

void __bc_error_print_impl(BCError* err, struct __bc_error_print_opts opts);

void bc_error_free(BCError* err);

#endif // BC_ERROR_H
