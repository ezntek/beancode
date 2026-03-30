/*
 * a_string/a_vector: a scuffed dynamic vector/string implementation.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */
#ifndef _A_STRING_SLICE_H
#define _A_STRING_SLICE_H

#include "common.h"

typedef struct {
    const char* data;
    usize len;
} a_string_slice;

#define astr_slice(s) ((a_string_slice){(s), strlen((s))});

#endif // A_STRING_SLICE_H
