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
#include "a_string.h"
#define _POSIX_C_SOURCE 200809L

#include <string.h>

#include "a_string_slice.h"
#include "error.h"

static const char* ERROR_KIND_STRINGS[] = {
    [BC_ERROR_EOF] = "EOFError",
    [BC_ERROR_RUNTIME] = "RuntimeError",
    [BC_ERROR_SYNTAX] = "SyntaxError",
};

static char error_kind_buf[32] = {0};

a_string_slice bc_error_kind_to_string_slice(BCErrorKind k) {
    strcpy(error_kind_buf, ERROR_KIND_STRINGS[k]);
    return astr_slice(error_kind_buf);
}

a_string bc_error_kind_to_string(BCErrorKind k) {
    return astr(ERROR_KIND_STRINGS[k]);
}

BCError bc_error_new(BCErrorKind k, BCPos p, a_string msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = msg,
    };
}

BCError bc_error_new_cstr(BCErrorKind k, BCPos p, const char* msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = astr(msg),
    };
}

BCError bc_error_new_string_slice(BCErrorKind k, BCPos p, a_string_slice msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = as_from_string_slice(msg),
    };
}
