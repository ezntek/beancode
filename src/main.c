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

#include "common.h"
#include "error.h"
#include "lexer.h"
#include "lexer_types.h"
#include "str.h"

i32 main(i32 argc, char **argv) {
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

    str file_content = {0};
    str_view file_name = sv_from_cstr(*argv);

    FILE *f = fopen(file_name.data, "r");
    if (!f)
        panic("could not open file %.*s", str_fmt(&file_name));
    file_content = str_read_entire_file(f);
    fclose(f);
    if (!str_is_valid(&file_content))
        panic("could not read file %.*s", str_fmt(&file_name));

    BCLexer l = bc_lexer_new(sv_from_str(&file_content));
    BCToken *tokens = NULL;
    usize len = bc_lexer_tokenize(&l, &tokens);

    if (!tokens) {
        bc_error_print(&l.error, file_name);
        bc_error_free(&l.error);
    } else {
        for (usize i = 0; i < len; i++) {
            str_view s = bc_token_to_string_slice_full(
                &tokens[i], sv_from_str(&file_content));
            eprintf("%.*s\n", str_fmt(&s));
        }
    }

    str_free(&file_content);
    free(tokens);

    return 0;
}
