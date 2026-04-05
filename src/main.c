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
#include "vm.h"
#include "vm_types.h"
#define _POSIX_C_SOURCE 200809L

#include <stddef.h>
#include <stdio.h>
// used in macro
#include <string.h>

#include "a_string.h"
#include "a_string_slice.h"
#include "common.h"
#include "error.h"
#include "lexer.h"
#include "lexer_types.h"

i32 main(i32 argc, char** argv) {
    argc--;
    argv++;

    /*
    char txt[] = "bogus amogus 3bd 3bc 3bb 3ba 3bz";
    char hello_world_buf[sizeof(txt) + sizeof(usize)] = {0};
    *(usize*)hello_world_buf = strlen(txt);
    strcpy(hello_world_buf + 8, txt);

    BCValue imms[] = {
        {.t = BC_TYPE_CHAR,    .v.c = '\n'               },
        {.t = BC_TYPE_STRING,  .v.s = hello_world_buf + 8},
        {.t = BC_TYPE_INTEGER, .v.i = 2                  },
    };

    // push '\n'
    // push "hello, world!"
    // output 2
    BCVM_Instr src[] = {
        (BC_INSTR_PUSH << 26) | 0,
        (BC_INSTR_PUSH << 26) | 1,
        (BC_INSTR_OUTPUT << 26) | 2,
    };
    BCVM vm = bc_vm_new(src, 3, imms, 3);
    bc_vm_exec(&vm);
    bc_vm_free(&vm);
    */

    return 0;

    a_string file_content = {0};
    a_string_slice file_name = {0};

    if (argc >= 1) {
        file_name = ass_from_cstr(*argv);
        file_content = as_read_file(file_name.data);
        if (!as_valid(&file_content))
            panic("could not read file %.*s", as_fmt(file_name));
    } else {
        file_name = ass_from_cstr("(stdin)");
        file_content = as_read_line(stdin);
        if (!as_valid(&file_content)) panic("could not read line from stdin");
    }

    BCLexer l = bc_lexer_new(ass_from_astr(file_content));
    BCToken* tokens = NULL;
    usize len = bc_lexer_tokenize(&l, &tokens);

    if (!tokens) {
        bc_error_print(&l.error, file_name);
        bc_error_free(&l.error);
    } else {
        for (usize i = 0; i < len; i++) {
            a_string_slice s = bc_token_to_string_slice_full(
                &tokens[i], ass_from_astr(file_content));
            eprintf("%.*s\n", as_fmt(s));
        }
    }

    as_free(&file_content);
    free(tokens);

    return 0;
}
