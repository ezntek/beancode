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

#include <stdio.h>
#include <string.h>

#include "common.h"
#include "error.h"
#include "str.h"

static const char *ERROR_KIND_STRINGS[] = {
    [BC_ERROR_EOF] = "EOFError",
    [BC_ERROR_RUNTIME] = "RuntimeError",
    [BC_ERROR_SYNTAX] = "SyntaxError",
};

static char error_kind_buf[32] = {0};

str_view bc_error_kind_to_string_slice(BCErrorKind k) {
    strcpy(error_kind_buf, ERROR_KIND_STRINGS[k]);
    return sv_from_cstr(error_kind_buf);
}

str bc_error_kind_to_string(BCErrorKind k) {
    return mstr(ERROR_KIND_STRINGS[k]);
}

BCError bc_error_new(BCErrorKind k, BCPos p, str msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = msg,
    };
}

BCError bc_error_new_cstr(BCErrorKind k, BCPos p, const char *msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = mstr(msg),
    };
}

BCError bc_error_new_string_slice(BCErrorKind k, BCPos p, str_view msg) {
    return (BCError){
        .kind = k,
        .pos = p,
        .msg = str_from_sv(msg),
    };
}

void bc_error_free(BCError *err) {
    str_free(&err->msg);
}

void __bc_error_print_impl(BCError *err, struct __bc_error_print_opts opts) {
    if (!opts.f)
        opts.f = stderr;

    if (opts.no_color) {
        fprintf(opts.f, "%.*s:%u:%u: error: ", str_fmt(&opts.file_name),
                err->pos.row, err->pos.col);
    } else {
        fprintf(opts.f, "\033[1m%.*s:%u:%u \033[31;1merror: \033[0m",
                str_fmt(&opts.file_name), err->pos.row, err->pos.col);
    }

    usize len = 0;

    for (; len < err->msg.len && err->msg.data[len] != '\n'; len++)
        continue;

    fprintf(opts.f, "%.*s\n", (int)len, err->msg.data);

    // TODO: print context

    if (sv_valid(&opts.src)) {
        panic("shawarma %d", 69);
        // TODO: source code printer
    }
}
