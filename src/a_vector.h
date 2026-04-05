/*
 * a_string/a_vector: a scuffed dynamic vector/string implementation.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */
#ifndef A_VECTOR_H
#define A_VECTOR_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#define AV_INITIAL_SIZE 4

#define AV_DECL(T, name)                                                       \
    typedef struct {                                                           \
        T* data;                                                               \
        usize len;                                                             \
        usize cap;                                                             \
    } name;

#define av_append(v, item)                                                     \
    do {                                                                       \
        if ((v)->len + 1 > (v)->cap) {                                         \
            if ((v)->cap == 0)                                                 \
                (v)->cap = AV_INITIAL_SIZE;                                    \
            else                                                               \
                (v)->cap += ((v)->cap >> 1);                                   \
            (v)->data = realloc((v)->data, sizeof(*(v)->data) * (v)->cap);     \
            check_alloc((v)->data);                                            \
        }                                                                      \
        (v)->data[(v)->len++] = (item);                                        \
    } while (0)

#define av_clear(v)                                                            \
    do {                                                                       \
        memset((v)->data, 0, sizeof(*(v)->data) * (v)->cap);                   \
        (v)->len = 0;                                                          \
    } while (0)

#define av_free(v)                                                             \
    do {                                                                       \
        if ((v)->data) {                                                       \
            free((v)->data);                                                   \
            (v)->data = NULL;                                                  \
        }                                                                      \
    } while (0)

#define av_reserve(v, amt)                                                     \
    do {                                                                       \
        if (amt >= (v)->len) {                                                 \
            (v)->data = realloc((v)->data, sizeof(*(v)->data) * amt);          \
            check_alloc((v)->data);                                            \
            (v)->cap = amt;                                                    \
        }                                                                      \
    } while (0)

#define av_append_many(v, itms, itms_len)                                      \
    do {                                                                       \
        if ((v)->len + itms_len > (v)->cap) {                                  \
            (v)->cap += itms_len;                                              \
            (v)->data = realloc((v)->data, sizeof(*(v)->data) * (v)->cap);     \
            check_alloc((v)->data);                                            \
        }                                                                      \
        memcpy(&(v)->data[(v)->len], itms, sizeof(*(v)->data) * itms_len);     \
        (v)->len += itms_len;                                                  \
    } while (0)

#define av_last(v) ((v)->data[(assert((v)->len > 0), (v)->len - 1)])

#define av_at(v, pos) ((v)->data[(assert(0 <= pos && pos < (v)->len), pos)])

#define av_pop(v) ((v)->data[(assert((v)->len > 0), --(v)->len)])

#define av_pop_many(v, count)                                                  \
    do {                                                                       \
        assert(count < (v)->len);                                              \
        (v)->len -= count;                                                     \
    } while (0)

#endif // A_VECTOR_H
