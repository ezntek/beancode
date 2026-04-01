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

#include <stdbool.h>

#include "common.h"

typedef struct {
    const char* data;
    usize len;
} a_string_slice;

#define ass_from_cstr(s)     ((a_string_slice){(s), strlen((s))})
#define ass_from_astr(s)     ((a_string_slice){(s).data, (s).len})
#define ass_from_astr_ptr(s) ((a_string_slice){(s)->data, (s)->len})

bool ass_valid(const a_string_slice s);

/**
 * gets the nth character from an a_string_slice.
 *
 * @param s the target string
 * @return the last character
 */
char ass_at(const a_string_slice s, usize idx);

/**
 * gets the first character from an a_string_slice.
 *
 * @param s the target string
 * @return the last character
 */
char ass_first(const a_string_slice s);

/**
 * gets the last character from an a_string_slice.
 *
 * @param s the target string
 * @return the last character
 */
char ass_last(const a_string_slice s);

bool ass_equal(const a_string_slice l, const a_string_slice r);

bool ass_equal_cstr(const a_string_slice l, const char* r);

bool ass_equal_nocase(const a_string_slice l, const a_string_slice r);

bool ass_equal_nocase_cstr(const a_string_slice l, const char* r);

bool ass_case_consistent(const a_string_slice s);

a_string_slice ass_substr(const a_string_slice s, usize begin, usize len);

#endif // A_STRING_SLICE_H
