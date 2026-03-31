/*
 * a_string/a_vector: a scuffed dynamic vector/string implementation.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */
#include <ctype.h>
#define _POSIX_C_SOURCE 200809L

// used in macro
#include <stdlib.h>

#include <string.h>
#include <strings.h>

#include "a_string_slice.h"

bool ass_valid(const a_string_slice s) {
    return s.data != NULL;
}

char ass_at(const a_string_slice s, usize idx) {
    if (!ass_valid(s)) panic("cannot operate on an invalid a_string_slice!");

    if (idx >= s.len)
        panic("a_string_slice index `%zu` out of range (length: `%zu`)", idx,
              s.len);

    return s.data[idx];
}

char ass_first(const a_string_slice s) {
    return ass_at(s, 0);
}

char ass_last(const a_string_slice s) {
    return ass_at(s, s.len - 1);
}

bool ass_equal(const a_string_slice l, const a_string_slice r) {
    return l.len == r.len && !strncmp(l.data, r.data, l.len);
}

bool ass_equal_cstr(const a_string_slice l, const char* r) {
    return l.len == strlen(r) && !strncmp(l.data, r, l.len);
}

bool ass_equal_nocase(const a_string_slice l, const a_string_slice r) {
    return l.len == r.len && !strncasecmp(l.data, r.data, l.len);
}

bool ass_equal_nocase_cstr(const a_string_slice l, const char* r) {
    return l.len == strlen(r) && !strncasecmp(l.data, r, l.len);
}

bool ass_case_consistent(const a_string_slice s) {
    if (!s.len) return true;

    int upper = isupper(s.data[0]);
    for (usize i = 1; i < s.len; i++) {
        if ((int)isupper(s.data[i]) != upper) return false;
    }

    return true;
}

a_string_slice ass_substr(const a_string_slice s, usize begin, usize len) {
    if (begin >= len)
        panic("a_string_slice substring begins at %zu, but the length of the "
              "string slice is %zu!",
              begin, s.len);

    if (begin + len >= s.len) len = s.len - begin;

    return (a_string_slice){
        .data = s.data + begin,
        .len = len,
    };
}
