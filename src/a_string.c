/*
 * a_string/a_vector: a scuffed dynamic vector/string implementation.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "a_string.h"
#include "a_vector.h"
#include "common.h"

a_string as_new(void) {
    return as_with_capacity(8);
}

a_string as_with_capacity(usize cap) {
    a_string res = {.len = 0, .cap = cap};

    res.data = calloc(res.cap, 1); // sizeof(char)
    check_alloc(res.data);

    return res;
}

void as_clear(a_string* restrict s) {
    memset(s->data, '\0', s->cap);
}

void as_free(a_string* restrict s) {
    if (!as_valid(s)) {
        return;
    }

    free(s->data);
    s->data = NULL;
    s->len = 0;
    s->cap = 0;
}

void as_copy(a_string* restrict dest, const a_string* restrict src) {
    if (!as_valid(dest)) panic("cannot operate on invalid a_string!");
    if (!as_valid(src)) panic("source string is invalid!");

    if (src->len > dest->cap) {
        as_reserve(dest, src->cap);
    }

    strcpy(dest->data, src->data);
    dest->len = src->len;
}

void as_copy_cstr(a_string* restrict dest, const char* src) {
    if (!as_valid(dest)) panic("cannot operate on invalid a_string!");
    if (src == NULL) panic("source C string is null!");

    usize len = strlen(src);
    if (len + 1 > dest->cap) {
        as_reserve(dest, len);
    }

    strncpy(dest->data, src, len);
    dest->len = len;
    if (dest->data[dest->len] != '\0') {
        dest->data[dest->len] = '\0'; // always nullterm
    }
}

void as_ncopy(a_string* restrict dest, const a_string* restrict src,
              usize chars) {
    if (!as_valid(dest)) panic("cannot operate on invalid a_string!");
    if (!as_valid(src)) panic("source string is invalid!");

    if (chars > dest->cap) {
        as_reserve(dest, chars);
    }
    as_clear(dest);

    stpncpy(dest->data, src->data, chars);
    dest->len = chars;
}

void as_ncopy_cstr(a_string* restrict dest, const char* src, usize chars) {
    if (!as_valid(dest)) panic("cannot operate on invalid a_string!");
    if (src == NULL) panic("source C string is null!");

    if (chars + 1 > dest->cap) {
        as_reserve(dest, chars + 1);
    }
    as_clear(dest);

    stpncpy(dest->data, src, chars);
    dest->len = chars;
}

void as_reserve(a_string* restrict s, usize cap) {
    if (!as_valid(s)) {
        panic("the string is invalid");
    }

    if (s->cap == cap) return;

    s->data = realloc(s->data, cap);
    check_alloc(s->data);
    s->cap = cap;
}

a_string as_from_cstr(const char* cstr) {
    if (cstr == NULL) panic("source C string is null!");

    a_string res = {
        .data = NULL,
        .cap = strlen(cstr) + 1,
        .len = strlen(cstr),
    };

    res.data = calloc(res.cap, 1); // sizeof(char)
    check_alloc(res.data);
    strcpy(res.data, cstr);

    return res;
}

a_string as_from_string_slice(a_string_slice slc) {
    a_string res = {.len = slc.len, .cap = slc.len + 1};

    res.data = calloc(res.cap, 1);
    check_alloc(res.data);
    strncpy(res.data, slc.data, slc.len);

    return res;
}

a_string astr(const char* cstr) {
    return as_from_cstr(cstr);
}

a_string as_dupe(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on invalid a_string!");

    a_string res = as_with_capacity(s->cap);
    res.len = s->len;
    strncpy(res.data, s->data, s->len);
    return res;
}

a_string as_asprintf(const char* restrict format, ...) {
    va_list args;
    va_start(args, format);

    va_list argscopy;
    va_copy(argscopy, args);

    usize len = vsnprintf(NULL, 0, format, argscopy);
    a_string res = as_with_capacity(len + 1);
    vsnprintf(res.data, res.cap, format, args);

    va_end(args);

    res.len = len;
    return res;
}

usize as_sprintf(a_string* restrict dest, const char* restrict format, ...) {
    va_list args;
    va_start(args, format);

    va_list argscopy;
    va_copy(argscopy, args);
    usize len = vsnprintf(NULL, 0, format, argscopy);

    if (as_valid(dest)) {
        as_reserve(dest, len + 1);
    } else {
        *dest = as_with_capacity(len + 1);
    }
    as_clear(dest);

    usize res = vsnprintf(dest->data, dest->cap, format, args);

    va_end(args);

    dest->len = len;
    return res;
}

int as_fprint(const a_string* restrict s, FILE* restrict stream) {
    return fprintf(stream, "%.*s", as_fmtp(s));
}

int as_fprintln(const a_string* restrict s, FILE* restrict stream) {
    return fprintf(stream, "%.*s\n", as_fmtp(s));
}

int as_print(const a_string* restrict s) {
    return as_fprint(s, stdout);
}

int as_println(const a_string* restrict s) {
    return as_fprintln(s, stdout);
}

char* as_fgets(a_string* restrict buf, usize cap, FILE* restrict stream) {
    usize actual_cap = (cap == 0) ? 8192 : cap;
    if (as_valid(buf)) {
        as_reserve(buf, actual_cap);
    } else {
        *buf = as_with_capacity(actual_cap);
    }
    char* fgets_res = fgets(buf->data, actual_cap, stream);
    if (fgets_res == NULL) return NULL;
    buf->len = strlen(buf->data);
    buf->data[buf->len] = 0;
    return buf->data;
}

a_string as_read_line(FILE* restrict stream) {
    a_string buf = as_new();

    char* res = as_fgets(&buf, 0, stream);
    if (res == NULL) return (a_string){0};

    // trim newline off
    if (as_last(&buf) == '\n') as_pop(&buf);

    // resize buf to appropriate size.
    as_reserve(&buf, buf.len + 1);
    return buf;
}

a_string as_read_file(const char* filename) {
    if (filename == NULL) panic("source file name C string is null!");

    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        return (a_string){0};
    }

    fseek(fp, 0, SEEK_END);
    usize sz = ftell(fp);
    rewind(fp);
    a_string res = as_with_capacity(sz + 1);
    if (fread(res.data, 1, sz, fp) != sz) {
        return (a_string){0};
    }
    res.len = sz;
    res.data[res.len] = '\0';
    fclose(fp);
    return res;
}

a_string as_input(const char* prompt) {
    if (prompt) {
        printf("%s", prompt);
        fflush(stdout);
    }

    a_string raw = as_read_line(stdin);
    if (!as_valid(&raw))
        return (a_string){0};
    else
        return raw;
}

bool as_valid(const a_string* restrict s) {
    return s->data != 0;
}

void as_append_char(a_string* restrict s, char c) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    if (s->len + 1 > s->cap) {
        as_reserve(s, s->cap * 2);
    }

    s->data[s->len++] = c;
}

void as_append_cstr(a_string* restrict s, const char* n) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    if (n == NULL) panic("null string passed to append operation!");

    usize new_len = strlen(n);
    usize required_cap = s->len + new_len + 1;
    if (required_cap > s->cap) {
        while (s->cap < required_cap) {
            as_reserve(s, s->cap * 2);
        }
    }

    for (usize i = 0; i < new_len; i++) {
        s->data[s->len++] = n[i];
    }
    s->data[s->len] = '\0'; // null terminate it
    s->len += new_len;
}

void as_append_astr(a_string* restrict s, const a_string* restrict n) {
    if (!as_valid(n)) panic("a_string to be appended cannot be NULL!");

    as_append_cstr(s, n->data);
}

void as_append(a_string* restrict s, const char* n) {
    as_append_cstr(s, n);
}

char as_pop(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    char last = s->data[--s->len];
    s->data[s->len] = '\0';
    return last;
}

char as_at(const a_string* restrict s, usize idx) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    if (idx >= s->len)
        panic("a_string index `%zu` out of range (length: `%zu`)!", idx,
              s->len);

    return s->data[idx];
}

char as_first(const a_string* restrict s) {
    return as_at(s, 0);
}

char as_last(const a_string* restrict s) {
    return as_at(s, s->len - 1);
}

a_string as_trim_left(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize i = 0;
    while (i < s->len) { // must check this first, else segfault
        if (strchr(" \n\t\r", s->data[i])) {
            i++;
        } else {
            break;
        }
    }

    return as_from_cstr(&s->data[i]);
}

a_string as_trim_right(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize end = s->len - 1;
    while (strchr(" \n\t\r", s->data[end])) { // end >= 0 is always true
        end--;
    }
    end++;

    a_string res = as_with_capacity(end + 1);
    for (usize i = 0; i < end; i++) {
        res.data[i] = s->data[i];
    }

    return res;
}

a_string as_trim(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize begin = 0;
    usize end = s->len - 1;

    while (begin < s->len) { // must check this first, else segfault
        if (strchr(" \n\t\r", s->data[begin]) != NULL) {
            begin++;
        } else {
            break;
        }
    }

    while (strchr(" \n\t\r", s->data[end])) { // end >= 0 is always true
        end--;
    }

    end++;

    a_string res = as_with_capacity(end - begin + 1);
    res.len = end - begin;

    usize i = 0;
    for (usize j = begin; j < end; j++) {
        res.data[i++] = s->data[j];
    }

    return res;
}

void as_inplace_trim_left(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize i = 0;
    while (i < s->len) { // must check this first, else segfault
        if (strchr(" \n\t\r", s->data[i])) {
            i++;
        } else {
            break;
        }
    }

    memmove(&s->data[0], &s->data[i], s->len - i);
}

void as_inplace_trim_right(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize end = s->len - 1;
    while (strchr(" \n\t\r", s->data[end])) { // end >= 0 is always true
        end--;
    }
    end++;

    usize oldlen = s->len;
    s->len = end + 1;

    if (end > s->len) memset(&s->data[end + 1], 0, oldlen - end + 1);
}

void as_inplace_trim(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    usize begin = 0;
    usize end = s->len - 1;

    while (begin < s->len) { // must check this first, else segfault
        if (isspace(s->data[begin])) {
            begin++;
        } else {
            break;
        }
    }

    while (isspace(s->data[end])) { // end >= 0 is always true
        end--;
    }

    end++;

    usize oldlen = s->len;
    s->len = end - begin;
    memmove(&s->data[0], &s->data[begin], s->len);
    if (end > s->len) memset(&s->data[end], 0, oldlen - s->len + 1);
}

a_string as_toupper(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    a_string res = as_with_capacity(s->len + 1);
    for (usize i = 0; i < s->len; i++) {
        res.data[i] = toupper(s->data[i]);
    }
    res.len = s->len;

    return res;
}

a_string as_tolower(const a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    a_string res = as_with_capacity(s->len + 1);
    for (usize i = 0; i < s->len; i++) {
        res.data[i] = tolower(s->data[i]);
    }
    res.len = s->len;

    return res;
}

void as_inplace_toupper(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    for (usize i = 0; i < s->len; i++) {
        s->data[i] = toupper(s->data[i]);
    }
}

void as_inplace_tolower(a_string* restrict s) {
    if (!as_valid(s)) panic("cannot operate on an invalid a_string!");

    for (usize i = 0; i < s->len; i++) {
        s->data[i] = tolower(s->data[i]);
    }
}

bool as_equal(const a_string* restrict lhs, const a_string* restrict rhs) {
    if (!as_valid(lhs)) panic("cannot compare an invalid a_string!");

    if (!as_valid(rhs)) panic("cannot compare an invalid a_string!");

    if (lhs->len != rhs->len) {
        return false;
    }

    for (usize i = 0; i < lhs->len; i++) {
        if (lhs->data[i] != rhs->data[i]) {
            return false;
        }
    }

    return true;
}

bool as_equal_cstr(const a_string* restrict lhs, const char* rhs) {
    if (!rhs) return false;
    a_string arhs = {
        .data = (char*)rhs,
        .len = strlen(rhs),
    }; // very sketchy, i know
    return as_equal(lhs, &arhs);
}

bool as_equal_case_insensitive(const a_string* restrict lhs,
                               const a_string* restrict rhs) {
    if (!as_valid(lhs)) panic("cannot compare an invalid a_string!");

    if (!as_valid(rhs)) panic("cannot compare an invalid a_string!");

    if (lhs->len != rhs->len) {
        return false;
    }

    for (usize i = 0; i < lhs->len; i++) {
        if (tolower(lhs->data[i]) != tolower(rhs->data[i])) {
            return false;
        }
    }

    return true;
}

bool as_equal_case_insensitive_cstr(const a_string* restrict lhs,
                                    const char* rhs) {
    if (!rhs) return false;
    a_string arhs = {
        .data = (char*)rhs,
        .len = strlen(rhs),
    }; // very sketchy, i know
    return as_equal_case_insensitive(lhs, &arhs);
}

a_string as_slice_cstr(const char* src, usize begin, usize end) {
    if (src == NULL) panic("source C string for slice operation is NULL!");

    if (begin > end)
        panic("begin cannot be greater than end in slice operation!");

    a_string res = as_new();
    as_ncopy_cstr(&res, &src[begin], end - begin);
    return res;
}

a_string as_slice(const a_string* restrict src, usize begin, usize end) {
    if (!as_valid(src)) panic("cannot slice an invalid a_string!");

    return as_slice_cstr(src->data, begin, end);
}

bool as_in(const a_string* restrict needle, const a_string** haystack,
           usize len) {
    for (usize i = 0; i < len; ++i) {
        if (as_equal(needle, haystack[i])) return true;
    }
    return false;
}

bool as_in_cstr(const a_string* restrict needle, const char** haystack,
                usize len) {
    for (usize i = 0; i < len; ++i) {
        if (as_equal_cstr(needle, haystack[i])) return true;
    }
    return false;
}

usize as_to_double(const a_string* restrict src, double* res) {
    errno = 0;
    char* endptr = NULL;
    const char* nptr = src->data;
    double dbl = strtod(nptr, &endptr);
    usize diff = endptr - nptr;
    if (diff == src->len) {
        *res = dbl;
        return diff;
    } else {
        return diff;
    }
}

usize as_to_integer(const a_string* restrict src, int64_t* res, int base) {
    errno = 0;
    char* endptr = NULL;
    const char* nptr = src->data;
    int64_t num = (int64_t)strtoll(nptr, &endptr, base);
    usize diff = endptr - nptr;
    if (diff == src->len) {
        *res = num;
        return diff;
    } else {
        return diff;
    }
}

#define iscont(c) (((u8)(c) & 0xC0) == 0x80)

usize au_len(const a_string* restrict s) {
    usize len = 0;
    for (usize i = 0; i < s->len; i++) {
        if (!iscont(s->data[i])) len++;
    }
    return len;
}

bool au_slice_valid(const u8* s, usize len) {
    if (!s) return false;

    usize rembytes = 0, saved_rembytes = 0, i = 0;
    u8 ch = 0;
    for (i = 0; i < len; i++) {
        ch = s[i];

        if (iscont(ch)) {
            if (!rembytes) return false; // stray continuation

            rembytes--;
            continue;
        }

        saved_rembytes = rembytes;
        if ((ch & 0x80) == 0) {
            rembytes = 0;
        } else if ((ch & 0xE0) == 0xC0) {
            rembytes = 1;
        } else if ((ch & 0xF0) == 0xE0) {
            rembytes = 2;
        } else if ((ch & 0xF8) == 0xF0) {
            rembytes = 3;
        } else {
            // junk
            return false;
        }

        if (saved_rembytes) return false;

        if (i + rembytes >= len) return false;

        if (rembytes) {
            u8 next = s[i + 1];
            if ((ch & 0xE0) == 0xC0) {
                // reject overlong, lead byte must be >=0b1100010
                if (ch < 0xC2) return false;
            } else if ((ch & 0xF0) == 0xE0) {
                // overlong
                if (ch == 0xE0 && next < 0xA0) return false;
                // reject surrogates
                if (ch == 0xED && next >= 0xA0) return false;
            } else if ((ch & 0xF8) == 0xF0) {
                // overlong
                if (ch == 0xF0 && next < 0x90) return false;
                // range
                if (ch == 0xF4 && next > 0x8F) return false;
                // cannot encode >U+10FFFF
                if (ch > 0xF4) return false;
            }
        }
    }

    if (rembytes) return false;

    return true;
}

bool au_valid(const a_string* restrict s) {
    if (!as_valid(s)) return false;

    return au_slice_valid((u8*)s->data, s->len);
}

u8* au_pos(const a_string* restrict s, usize idx) {
    usize cur_cp = 0;

    for (usize i = 0; i < s->len; i++) {
        if (!iscont(s->data[i])) {
            if (cur_cp == idx) return (u8*)&s->data[i];
            cur_cp++;
        }
    }

    return NULL;
}

u8* au_next_begin(const a_string* restrict s, u8* begin) {
    if (!begin) begin = (u8*)s->data;

    const u8* endptr = (u8*)s->data + s->len;

    // sync first
    while (begin < endptr && iscont(*begin)) {
        begin++;
    }
    // we are now on a leader
    begin++;

    if (begin >= endptr) return NULL;

    // sync to the next one
    while (begin < endptr && iscont(*begin)) {
        begin++;
    }
    if (begin >= endptr) return NULL;

    return begin;
}

dchar au_decode(u8* ptr) {
    if (iscont(*ptr)) panic("cannot decode from a continuation byte!");

    dchar res = 0;
    u8 initial = ptr[0];
    if ((initial & 0x80) == 0) {
        res = initial;
    } else if ((initial & 0xE0) == 0xC0) {
        res = (dchar)((initial & 0x1F) << 6) | (ptr[1] & 0x3F);
    } else if ((initial & 0xF0) == 0xE0) {
        res = (dchar)((initial & 0x0F) << 12) | (ptr[1] & 0x3F) << 6 |
              (ptr[2] & 0x3F);
    } else if ((initial & 0xF8) == 0xF0) {
        res = (dchar)((initial & 0x07) << 18) | (ptr[1] & 0x3F) << 12 |
              (ptr[2] & 0x3F) << 6 | (ptr[3] & 0x3F);
    } else {
        panic("junk found in UTF-8 string");
    }

    return res;
}

dchar au_next_codepoint(const a_string* restrict s, u8* begin) {
    u8* res = au_next_begin(s, begin);
    if (!res)
        panic("tried to get next character of a UTF-8 string, but it is out of "
              "bounds");

    return au_decode(res);
}

dchar au_at(const a_string* restrict s, usize idx) {
    u8* pos = au_pos(s, idx);

    if (!pos) panic("character at position %d out of bounds!", (int)idx);

    if (iscont(*pos))
        panic("tried to get a unicode character from a continuation byte!");

    return au_decode(pos);
}

a_dstring au_codepoints(const a_string* restrict s) {
    a_dstring res = {0};

    au_iter(s, ptr) av_append(&res, au_decode(ptr));

    return res;
}

u8 au_encode_cp(u8 dest[4], dchar src) {
    if (src > 0x10FFFF || (0xD800 <= src && src <= 0xDFFF)) {
        panic("invalid codepoint U+`%X` in dchar!", src);
    }

    memset((void*)dest, 0, 4);
    usize len = 1 + (src > 0x7F) + (src > 0x7FF) + (src > 0xFFFF);
    // write continuation bytes in reverse
    for (usize i = len - 1; i > 0; i--) {
        dest[i] = 0x80 | (src & 0x3F);
        src >>= 6;
    }

    static const u8 mask[5] = {0x00, 0x00, 0xC0, 0xE0, 0xF0};
    dest[0] = mask[len] | (u8)src;
    return len;
}

void au_append_char(a_string* restrict s, dchar cp) {
    u8 buf[4], count = 0;
    count = au_encode_cp(buf, cp);

    if (s->cap + count > s->cap) as_reserve(s, s->cap + count);

    memcpy(&s->data[s->len], buf, count);
    s->len += count;
    s->data[s->len] = 0;
}

void au_append_slice(a_string* restrict s, const u8* data, usize len) {
    if (!au_slice_valid(data, len))
        panic("tried to append invalid UTF-8 slice to a_string!");

    if (s->cap + len > s->cap) as_reserve(s, s->cap + len);

    memcpy(&s->data[s->len], data, len);
    s->len += len;
    s->data[s->len] = 0;
}

void au_append_astr(a_string* restrict s, const a_string* restrict other) {
    if (!as_valid(other))
        panic("tried to append invalid UTF-8 a_string to a_string!");

    au_append_slice(s, (u8*)other->data, other->len);
}
