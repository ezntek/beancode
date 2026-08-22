/*
 * ezntek's str.h: a simple string/string slice library for standard C99.
 *
 * Copyright (c) Eason Qin, 2026.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef EZNTEK_STR_H
#define EZNTEK_STR_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * Heap-allocated, owned character string.
 *
 * All library functions assert that pointers passed in are valid and point to
 * valid data. All references to "C strings" infer null-terminated pointers to
 * characters.
 *
 * When data is NULL, it is deemed invalid.
 */
typedef struct {
    char *data;
    size_t cap;
    size_t len;
} str;

/**
 * A view into any character buffer that may not be
 * resized, but may be heap-allocated.
 *
 * When data is NULL, it is deemed invalid.
 */
typedef struct {
    const char *data;
    size_t len;
} str_view;

#define str_is_valid(str) ((str)->data != NULL)

// for printf formatting
#define str_fmt(s) (int)((s)->len), (s)->data

/**
 * Creates and initializes an empty, valid str.
 */
str str_new(void);

/**
 * Creates and initializes an empty str with a specified capacity.
 */
str str_new_with_capacity(size_t cap);

/**
 * Make a str from a C string.
 */
str str_from_cstr(const char *s);

/**
 * Adopts an existing heap-allocated C string, reusing its buffer in the
 * resultant str.
 */
str str_adopt(char *s);

/**
 * Equivalent to str_from_cstr
 */
str mstr(const char *s);

/**
 * Make a str from a string slice.
 */
str str_from_sv(const str_view s);

/**
 * Return a str from the result of an sprintf-style call.
 */
str str_format(const char *format, ...);

/**
 * Saves the result of an sprintf-style call into s.
 */
void str_format_into(str *s, const char *format, ...);

/**
 * Reserves 1.5*new_cap bytes of space in the string.
 */
void str_reserve(str *s, size_t new_cap);

/**
 * Reserves EXACTLY new_cap of bytes of space in the string.
 */
void str_reserve_exact(str *s, size_t new_cap);

/**
 * Clears the entire string buffer to zeroes and resets the length.
 */
void str_clear(str *s);

/**
 * Frees the internal heap-allocated buffer.
 */
void str_free(str *s);

/**
 * Copies the entirety of src into dest, resizing it if necessary, while
 * overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_copy_into(str *dest, const str *src);

/**
 * Copies a source C string at src into an already initialized
 * str, resizing the buffer if necessary, overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_copy_cstr_into(str *dest, const char *src);

/**
 * Copies a source string slice at src into an already initialized
 * str, resizing the buffer if necessary, overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_copy_sv_into(str *dest, const str_view src);

/**
 * Copies n bytes of src into an already initialized str dest,
 * resizing it if necessary, while overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_ncopy_into(str *dest, const str *src, size_t count);

/**
 * Copies n bytes of a C string src into an already initialized str dest,
 * resizing it if necessary, while overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_ncopy_cstr_into(str *dest, const char *src, size_t count);

/**
 * Copies n bytes of a string slice src into an already initialized str dest,
 * resizing it if necessary, while overwriting all existing data.
 * dest will be initialized if necessary.
 */
void str_ncopy_sv_into(str *dest, const str_view src, size_t count);

/**
 * Duplicates a str exactly, retaining its original capacity.
 */
str str_dupe(const str *src);

/**
 * Appends a str into an existing str, resizing the buffer if necessary.
 */
void str_append(str *src, const str *new_str);

/**
 * Appends a single character new into src, resizing the buffer if necessary.
 */
void str_append_char(str *src, char new_char);

/**
 * Appends a C string into an existing str, resizing the buffer if necessary.
 */
void str_append_cstr(str *src, const char *new_cstr);

/**
 * Appends a string slice into an existing str, resizing the buffer if
 * necessary.
 */
void str_append_sv(str *src, const str_view new_slc);

/**
 * Appends a new formatted section of the string, with the same calling
 * style as str_append into an existing string buffer.
 */
void str_append_format(str *dest, const char *format, ...);

/**
 * Removes the last character from src, returning it.
 */
char str_pop_last(str *src);

/**
 * Concatenates lhs and rhs together into a new string.
 */
str str_concat(const str *lhs, const str *rhs);

/**
 * Concatenates lhs and a C string rhs together into a new string.
 */
str str_concat_cstr(const str *lhs, const char *rhs);

/**
 * Concatenates lhs and a str_view rhs together into a new string.
 */
str str_concat_sv(const str *lhs, const str_view rhs);

/**
 * Trims all whitespace characters in s away only on the left, and returns the
 * result.
 */
str str_trim_left(const str *s);

/**
 * Trims all whitespace characters in s away only on the left, and returns the
 * result.
 */
str str_trim_right(const str *s);

/**
 * Trims all whitespace characters in s away on the left and right, and returns
 * the result.
 */
str str_trim(const str *s);

/**
 * Converts all characters in s to uppercase and returns the result.
 */
str str_to_upper(const str *s);

/**
 * Converts all characters in s to lowercase and returns the result
 */
str str_to_lower(const str *s);

/**
 * Trims all whitespace characters in s away only on the left, modifying the
 * string s.
 */
void str_trim_left_in_place(str *s);

/**
 * Trims all whitespace characters in s away only on the right, modifying the
 * string s.
 */
void str_trim_right_in_place(str *s);

/**
 * Trims all whitespace characters in s away on the left and right, modifying
 * the string s.
 */
void str_trim_in_place(str *s);

/**
 * Converts all characters in s to uppercase, modifying the string s.
 */
void str_to_upper_in_place(str *s);

/**
 * Converts all characters in s to lowercase, modifying the string s.
 */
void str_to_lower_in_place(str *s);

/**
 * Returns if lhs and rhs are equal.
 */
bool str_equal(const str *lhs, const str *rhs);

/**
 * Returns if lhs and rhs (a C string) are equal.
 */
bool str_equal_cstr(const str *lhs, const char *rhs);

/**
 * Returns if lhs and rhs (a string slice) are equal.
 */
bool str_equal_sv(const str *lhs, const str_view rhs);

/**
 * Compares the strings lhs and rhs lexicographically, similar to strcmp.
 */
int str_compare(const str *lhs, const str *rhs);

/**
 * Compares the strings lhs and rhs (a C string) lexicographically, similar to
 * strcmp.
 */
int str_compare_cstr(const str *lhs, const char *rhs);

/**
 * Compares the strings lhs and rhs (a string slice) lexicographically, similar
 * to strcmp.
 */
int str_compare_sv(const str *lhs, const str_view rhs);

/**
 * Returns if lhs and rhs are equal, ignoring case.
 */
bool str_equal_case_insensitive(const str *lhs, const str *rhs);

/**
 * Returns if lhs and rhs (a C string) are equal, ignoring case.
 */
bool str_equal_cstr_case_insensitive(const str *lhs, const char *rhs);

/**
 * Returns if lhs and rhs (a string slice) are equal, ignoring case.
 */
bool str_equal_sv_case_insensitive(const str *lhs, const str_view rhs);

/**
 * Compares the strings lhs and rhs lexicographically, ignoring case, similar to
 * strcasecmp.
 */
int str_compare_case_insensitive(const str *lhs, const str *rhs);

/**
 * Compares the strings lhs and rhs (a C string) lexicographically, ignoring
 * case, similar to strcasecmp.
 */
int str_compare_cstr_case_insensitive(const str *lhs, const char *rhs);

/**
 * Compares the strings lhs and rhs (a String slice) lexicographically, ignoring
 * case, similar to strcasecmp.
 */
int str_compare_sv_case_insensitive(const str *lhs, const str_view rhs);

/**
 * Slices a string from `begin` to `end`, clamping values to [0, src->len] if
 * needed, and returning the result.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str str_slice_to_str(const str *src, ptrdiff_t begin, ptrdiff_t end);

/**
 * Slices a C string from `begin` to `end`, clamping values to [0, src->len] if
 * needed, returning the result as a str.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str str_slice_from_cstr(const char *src, ptrdiff_t begin, ptrdiff_t end);

/**
 * Slices a string from `begin` to `end`, clamping values to [0, src->len] if
 * needed, returning the result as a str_view.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str_view str_slice(const str *src, ptrdiff_t begin, ptrdiff_t end);

/**
 * Slices a string slice from beginning to end, clamping values to [0, src->len]
 * if needed Python, returning the result as a str.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str str_slice_from_sv(const str_view src, ptrdiff_t begin, ptrdiff_t end);

/**
 * Converts a str to a double with strtod(3), returning how many bytes of src
 * were converted to a double.
 */
size_t str_to_double_pro(const str *src, double *res);

/**
 * Converts a str to a int64_t with strtoll(3), returning how many bytes of src
 * were converted to a double.
 */
size_t str_to_int64_pro(const str *src, int64_t *res, int base);

/**
 * Converts a str to a double with strtod(3), returning a success or failure.
 */
bool str_to_double(const str *src, double *res);

/**
 * Converts a str to a int64_t with strtoll(3), returning a success or failure.
 */
bool str_to_int64(const str *src, int64_t *res);

/**
 * Reads a single line from f into a str, and returns it.
 * Panics on failure.
 */
str str_read_line_from_file(FILE *f);

/**
 * Reads count characters from f into a str, and returns it.
 * Panics on failure.
 */
str str_read_chars_from_file(FILE *f, size_t count);

/**
 * Reads the entirety of f into a str, and returns it.
 * Panics on failure.
 */
str str_read_entire_file(FILE *f);

/**
 * Writes the entirety of a str to a file, returning a success or failure.
 * Panics on failure.
 */
bool str_write_to_file(const str *s, FILE *f);

/**
 * Gets the nth character of a str, with bounds checking.
 */
char str_at(const str *s, size_t i);

/**
 * Gets the first character of a str, with bounds checking.
 */
char str_first(const str *s);

/**
 * Gets the last character of a str, with bounds checking.
 */
char str_last(const str *s);

/**
 * Finds a character c in s, from the left to the right.
 * Returns -1 on failure
 */
ptrdiff_t str_find_char(const str *s, char c);

/**
 * Finds a character c in s from the right to the left.
 * Returns s->len on failure.
 */
ptrdiff_t str_find_char_right(const str *s, char c);

bool str_case_consistent(const str *s);

#define sv_valid(sv) ((sv)->data != NULL)

void str_print(const str *s);

void str_println(const str *s);

void str_fprint(const str *s, FILE *f);

void str_fprintln(const str *s, FILE *f);

/**
 * Converts a str to a str_view.
 */
str_view sv_from_str(const str *s);

/**
 * Converts a C string to a str_view.
 */
str_view sv_from_cstr(const char *s);

/**
 * Equivalent to sv_from_cstr.
 */
str_view msv(const char *s);

str sv_concat(const str_view lhs, const str_view rhs);

str sv_concat_cstr(const str_view lhs, const char *rhs);

bool sv_equal(const str_view lhs, const str_view rhs);

bool sv_equal_cstr(const str_view lhs, const char *rhs);

bool sv_equal_case_insensitive(const str_view lhs, const str_view rhs);

bool sv_equal_cstr_case_insensitive(const str_view lhs, const char *rhs);

int sv_compare(const str_view lhs, const str_view rhs);

int sv_compare_cstr(const str_view lhs, const char *rhs);

int sv_compare_case_insensitive(const str_view lhs, const str_view rhs);

int sv_compare_cstr_case_insensitive(const str_view lhs, const char *rhs);

bool sv_case_consistent(const str_view sv);

/**
 * Slices a string slice from `begin` to `end`, clamping values to [0, src->len]
 * if needed, and returning the result.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str_view sv_slice(const str_view s, ptrdiff_t begin, ptrdiff_t end);

/**
 * Slices a string slice from `begin` to `end`, clamping values to [0, src->len]
 * if needed, and returning the result.
 * ptrdiff_t's are used, as slicing functions may return negative values.
 */
str sv_slice_to_str(const str_view s, ptrdiff_t begin, ptrdiff_t end);

str sv_to_lower(const str_view s);
str sv_to_upper(const str_view s);

str_view sv_trim_left(const str_view s);
str_view sv_trim_right(const str_view s);
str_view sv_trim(const str_view s);

size_t sv_to_double_pro(const str_view src, double *res);
size_t sv_to_int64_pro(const str_view src, int64_t *res, int base);
bool sv_to_double(const str_view src, double *res);
bool sv_to_int64(const str_view src, int64_t *res);

bool sv_write_to_file(const str_view src, FILE *f);

char sv_first(const str_view s);
char sv_last(const str_view s);
char sv_at(const str_view s, size_t i);

void sv_print(const str_view s);
void sv_println(const str_view s);
void sv_fprint(const str_view s, FILE *f);
void sv_fprintln(const str_view s, FILE *f);

/**
 * Finds a character c in s, from the left to the right.
 * Returns -1 on failure
 */
ptrdiff_t sv_find_char(const str_view s, char c);

/**
 * Finds a character c in s from the right to the left.
 * Returns s->len on failure.
 */
ptrdiff_t sv_find_char_right(const str_view s, char c);

str str_from_after_delim(str *arg, char delimiter);
str str_from_after_delim_sv(str_view arg, char delimiter);
str str_from_before_delim(str *arg, char delimiter);
str str_from_before_delim_sv(str_view arg, char delimiter);
str_view sv_from_after_delim(str_view arg, char delimiter);
str_view sv_from_before_delim(str_view arg, char delimiter);

#endif
