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
// used in macro
#include <string.h>

#include "a_string.h"
#include "a_string_slice.h"
#include "common.h"

i32 main(i32 argc, char** argv) {
    argc--;
    argv++;

    a_string file_content = {0};
    a_string_slice file_name = {0};

    if (argc >= 1) {
        file_name = astr_slice(*argv);
        file_content = as_read_file(file_name.data);
        if (!as_valid(&file_content)) panic("could not read file %.*s", as_fmt(file_name));
    } else {
        file_name = astr_slice("(stdin)");
        file_content = as_read_line(stdin);
        if (!as_valid(&file_content)) panic("could not read line from stdin");
    }

    return 0;
}
