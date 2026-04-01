/*
 * common typedefs and DEFINEs to make life easier
 * (used by a_string and a_vector)
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */

#ifndef _A_COMMON_H
#define _A_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

typedef size_t usize;
typedef ssize_t isize;
typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;
typedef int64_t i64;
typedef float f32;
typedef double f64;

#define S_BOLD "\033[1m"
#define S_DIM  "\033[2m"
#define S_END  "\033[0m"

#define S_BLACK   "\033[30m"
#define S_RED     "\033[31m"
#define S_GREEN   "\033[32m"
#define S_YELLOW  "\033[33m"
#define S_BLUE    "\033[34m"
#define S_MAGENTA "\033[35m"
#define S_CYAN    "\033[36m"
#define S_WHITE   "\033[37m"

#define S_BG_BLACK   "\033[40m"
#define S_BG_RED     "\033[41m"
#define S_BG_GREEN   "\033[42m"
#define S_BG_YELLOW  "\033[43m"
#define S_BG_BLUE    "\033[44m"
#define S_BG_MAGENTA "\033[45m"
#define S_BG_CYAN    "\033[46m"
#define S_BG_WHITE   "\033[47m"

#define S_CLEAR_SCREEN "\033[2J\033[H"
#define S_CLEAR_LINE   "\r\033[K"

#define S_ENTER_ALT "\033[?1049h"
#define S_LEAVE_ALT "\031[?1049l"

#define S_SHOWCURSOR "\033[?25h"
#define S_HIDECURSOR "\033[?25l"

#define LENGTH(lst) (i32)(sizeof(lst) / sizeof(lst[0]))
#define RINTC(T)    *(T*)&

#define check_alloc(ptr)                                                       \
    do {                                                                       \
        if (!ptr) {                                                            \
            panic("allocation of `%s` failed", #ptr);                          \
            perror("perror");                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define eprintf(...) fprintf(stderr, __VA_ARGS__)

#define panic(...)                                                             \
    do {                                                                       \
        eprintf("\033[31;1mpanic:\033[0m line %d, func \"%s\" in file "        \
                "\"%s\": ",                                                    \
                __LINE__, __func__, __FILE__);                                 \
        eprintf(__VA_ARGS__);                                                  \
        eprintf("\n");                                                         \
        abort();                                                               \
    } while (0)

#ifdef unreachable
#undef unreachable
#endif
#define unreachable panic("reached unreachable code")

#define fatal_noexit(...)                                                      \
    do {                                                                       \
        eprintf(S_RED S_BOLD "[fatal] " S_END);                                \
        eprintf(S_DIM);                                                        \
        eprintf(__VA_ARGS__);                                                  \
        eprintf(S_END "\n");                                                   \
    } while (0)

#define fatal(...)                                                             \
    do {                                                                       \
        fatal_noexit(__VA_ARGS__);                                             \
        exit(1);                                                               \
    } while (0);

#define warn(...)                                                              \
    do {                                                                       \
        eprintf(S_MAGENTA S_BOLD "[warn] " S_END);                             \
        eprintf(S_DIM);                                                        \
        eprintf(__VA_ARGS__);                                                  \
        eprintf(S_END "\n");                                                   \
    } while (0)

#define info(...)                                                              \
    do {                                                                       \
        eprintf(S_CYAN S_BOLD "[info] " S_END);                                \
        eprintf(S_DIM);                                                        \
        eprintf(__VA_ARGS__);                                                  \
        eprintf(S_END "\n");                                                   \
    } while (0)

#define make(T, ident, val)                                                    \
    do {                                                                       \
        (ident) = malloc(sizeof(T));                                           \
        check_alloc((ident));                                                  \
        *(ident) = (val);                                                      \
    } while (0)

#define if_let(type, id, expr)                                                 \
    type id;                                                                   \
    if ((id = (expr).data, (expr)).have)

#define let_else(type, id, expr)                                               \
    type id;                                                                   \
    if (!(id = (expr).data, (expr)).have)

#define while_let(type, id, expr)                                              \
    type id;                                                                   \
    while ((id = (expr).data, (expr)).have)

#define let(id, expr) ((id = (expr).data, (expr)).have)

#endif // _A_COMMON_H
