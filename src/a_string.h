/*
 * a_string/a_vector: a scuffed dynamic vector/string implementation.
 *
 * Copyright (c) Eason Qin, 2025-2026.
 *
 * This source code form is licensed under the MIT/Expat license.
 * Visit the OSI website for a digital version.
 */
#ifndef A_STRING_H
#define A_STRING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "a_string_slice.h"
#include "a_vector.h"
#include "common.h"

/// wide char
typedef u16 wchar;

/// double wide char
typedef u32 dchar;

#define as_fmt(s)  (int)((s).len), ((s).data)
#define as_fmtp(s) (int)((s)->len), ((s)->data)

/**
 * null terminated, heap-allocated string slice.
 */
typedef struct {
    // The raw string slice allocated on the heap.
    char* data;

    // length of the string.
    usize len;

    // capacity of the string. includes the null terminator.
    usize cap;
} a_string;

/**
 * creates and initializes an empty, valid a_string. If you would like to create
 * an uninitialized and invalid a_string, use `as_new_uninitialized`.
 */
a_string as_new(void);

/**
 * creates an empty a_string with a specified capacity.
 *
 * @param cap the capacity of the string.
 */
a_string as_with_capacity(usize cap);

/**
 * clears the string with null terminators, keeping the capacity.
 *
 * @param s the string
 */
void as_clear(a_string* restrict s);

/**
 * destroys the heap-allocated data in an a_string. Do not read from the
 * a_string after it is destroyed!
 *
 * if the a_string is invalid, this is a no-op.
 *
 * @param s the string to be destroyed
 */
void as_free(a_string* restrict s);

/**
 * Copies one a_string to another.
 *
 * @param dest the dest string
 * @param src the source string
 */
void as_copy(a_string* restrict dest, const a_string* restrict src);

/**
 * Copies the entirety of a C string into an a_string.
 *
 * Passing a null-terminated string causes undefined behavior.
 *
 * @param dest the dest string
 * @param src the source string
 */
void as_copy_cstr(a_string* restrict dest, const char* src);

/**
 * Copies N bytes of one a_string to another.
 *
 * @param dest the dest string
 * @param src the source string
 * @param chars the number of chars
 */
void as_ncopy(a_string* restrict dest, const a_string* restrict src, usize chars);

/**
 * Copies N bytes of one C string to an a_string.
 *
 * @param dest the dest string
 * @param src the source string
 * @param chars the number of chars
 */
void as_ncopy_cstr(a_string* restrict dest, const char* src, usize chars);

/**
 * reserves a specific capacity on an a_string.
 *
 * @param s the string to be modified
 * @param cap the new capacity of the string
 */
void as_reserve(a_string* restrict s, usize cap);

/**
 * creates an a_string from a C string.
 *
 * @param cstr the C string to be converted. This function does not free the
 * string if it is heap-allocated, the string is instead duplicated.
 */
a_string as_from_cstr(const char* cstr);

/**
 * creates an a_string from an a_string_slice.
 */
a_string as_from_string_slice(a_string_slice slc);

/**
 * creates an a_string from a C string.
 *
 * shorthand of `as_from_cstr()`
 *
 * @param cstr the C string to be converted. This function does not free the
 * string if it is heap-allocated, the string is instead duplicated.
 */
a_string astr(const char* cstr);

/**
 * duplicates an a_string.
 *
 * the exact string, including the capacity it holds will be duplicated.
 *
 * @param s the string to duplicate
 */
a_string as_dupe(const a_string* restrict s);

/**
 * similar to asprintf, but for an a_string.
 *
 * @param format the format
 * @param ... format args
 */
a_string as_asprintf(const char* format, ...);

/**
 * similar to sprintf, but for an a_string.
 *
 * The string is guaranteed to be null-terminated. Passing in a valid a_string
 * will result in its buffer being overwritten by the formatted data, with its
 * capacity resized accordingly.
 *
 * Passing in an uninitialized/invalid a_string will create a new one.
 *
 * @param format the format
 * @param ... format args
 */
usize as_sprintf(a_string* restrict dest, const char* format, ...);

/**
 * prints an a_string to a file stream.
 *
 * @param s the string
 * @param stream the stream
 * @return number of bytes written
 */
int as_fprint(const a_string* restrict s, FILE* stream);

/**
 * prints an a_string to a file stream, with a newline.
 *
 * @param s the string
 * @param stream the stream
 * @return number of bytes written
 */
int as_fprintln(const a_string* restrict s, FILE* stream);

/**
 * prints an a_string to stdout.
 *
 * @param s the string
 * @param stream the stream
 * @return number of bytes written
 */
int as_print(const a_string* restrict s);

/**
 * prints an a_string to stdout, with a newline.
 *
 * @param s the string
 * @param stream the stream
 * @return number of bytes written
 */
int as_println(const a_string* restrict s);

/**
 * similar to fgets, but reads into an a_string. it returns the
 * buffer at buf.data or NULL.
 *
 * The string is guaranteed to be null-terminated. Passing in a valid a_string
 * will result in its buffer being overwritten with a buffer of capacity cap.
 * passing in a capacity of 0 will result in a default capacity of 8192.
 *
 * @param cap the maximum capacity of the string to be entered.
 * @param stream the file stream.
 */
char* as_fgets(a_string* restrict buf, usize cap, FILE* stream);

/**
 * reads a single line from a file.
 *
 * the resultant string's size is resized to the number of characters read, if
 * less than 8192 bytes.
 *
 * @param buf the target buffer to write into, it can be either valid or invalid
 * @param stream the target file stream
 * @return invalid string on failure
 */
a_string as_read_line(FILE* stream);

/**
 * reads the entirety of a file into an a_string.
 *
 * the capacity of the string will be equal to the length of string read from
 * the file. the maximum supported capacity for a single line is 8192 chars.
 *
 * Returns an invalid `a_string` upon error, and sets errno according to fopen.
 *
 * @param filename the name of the file.
 * @return invalid string on failure
 */
a_string as_read_file(const char* filename);

/**
 * gets a string input from stdin into an a_string with a non-formatted prompt.
 *
 * only up to 8192 characters may be read.
 *
 * @param prompt a C string to be printed as the prompt. leaving it as null will
 *               print no prompt.
 */
a_string as_input(const char* prompt);

/**
 * checks if an a_string is valid
 *
 * @param s the string to be checked
 */
bool as_valid(const a_string* restrict s);

/**
 * adds 1 character to an a_string
 *
 * @param s the target string to be concatenated
 * @param c the character to be added.
 */
void as_append_char(a_string* restrict s, char c);

/**
 * concatenates 2 a_strings together.
 *
 * @param s the target string to be concatenated
 * @param new the string to add on. This string will be kept intact and will not
 * be freed.
 */
void as_append_astr(a_string* restrict s, const a_string* restrict n);

/**
 * concatenates a C string to an a_string.
 *
 * @param s the target string to be concatenated
 * @param new the string to add on. This is a C string that will be kept intact.
 */
void as_append_cstr(a_string* restrict s, const char* n);

/**
 * concatenates a C string to an a_string.
 *
 * shorthand form of `as_append_cstr()`.
 *
 * @param new the string to add on. This string will be kept intact and will not
 * be freed.
 */
void as_append(a_string* restrict s, const char* n);

/**
 * removes the last character from an a_string.
 *
 * @param s the target string
 * @return the last character
 */
char as_pop(a_string* restrict s);

/**
 * gets the nth character from an a_string.
 *
 * @param s the target string
 * @return the last character
 */
char as_at(const a_string* restrict s, usize idx);

/**
 * gets the first character from an a_string.
 *
 * @param s the target string
 * @return the last character
 */
char as_first(const a_string* restrict s);

/**
 * gets the last character from an a_string.
 *
 * @param s the target string
 * @return the last character
 */
char as_last(const a_string* restrict s);

/**
 * removes all whitespace characters from the left side of an a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
a_string as_trim_left(const a_string* restrict s);

/**
 * removes all whitespace characters from the right side of an a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
a_string as_trim_right(const a_string* restrict s);

/**
 * removes all whitespace characters from the left and right sides of an
 * a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
a_string as_trim(const a_string* restrict s);

/**
 * (IN PLACE) removes all whitespace characters from the left side of an
 * a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
void as_inplace_trim_left(a_string* restrict s);

/**
 * (IN PLACE) removes all whitespace characters from the right side of an
 * a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
void as_inplace_trim_right(a_string* restrict s);

/**
 * (IN PLACE) removes all whitespace characters from the left and right sides of
 * an a_string.
 *
 * @param s the string
 * @return a new a_string.
 */
void as_inplace_trim(a_string* restrict s);

/**
 * converts all the characters in the a_string to uppercase.
 *
 * @param s the string
 */
a_string as_toupper(const a_string* restrict s);

/**
 * converts all the characters in the a_string to lowercase.
 *
 * @param s the string
 */
a_string as_tolower(const a_string* restrict s);

/**
 * (IN PLACE) converts all the characters in the a_string to uppercase.
 *
 * @param s the string
 */
void as_inplace_toupper(a_string* restrict s);

/**
 * (IN PLACE) converts all the characters in the a_string to lowercase.
 *
 * @param s the string
 */
void as_inplace_tolower(a_string* restrict s);

/**
 * checks if 2 a_strings are the same.
 *
 * @param lhs the first string
 * @param rhs the other string
 */
bool as_equal(const a_string* restrict lhs, const a_string* restrict rhs);

/**
 * checks if an a_string is equal to a C string, case insensitive
 *
 * @param lhs the first string
 * @param rhs the other string
 */
bool as_equal_cstr(const a_string* restrict lhs, const char* rhs);

/**
 * checks if 2 a_strings are the same, case insensitive.
 *
 * @param lhs the first string
 * @param rhs the other string
 */
bool as_equal_case_insensitive(const a_string* restrict lhs, const a_string* restrict rhs);

/**
 * checks if an a_string is equal to a C string, case insensitive
 *
 * @param lhs the first string
 * @param rhs the other string
 */
bool as_equal_case_insensitive(const a_string* restrict lhs, const a_string* restrict rhs);

/**
 * slices an a_string from begin to end, from a C string, discluding end.
 *
 * @param src the source string
 * @param begin the beginning
 * @param end the end
 * @return the new string
 */
a_string as_slice_cstr(const char* src, usize begin, usize end);

/**
 * slices an a_string from begin to end, discluding end.
 *
 * @param src the source string
 * @param begin the beginning
 * @param end the end
 * @return the new string
 */
a_string as_slice(const a_string* restrict src, usize begin, usize end);

/**
 * checks if a target string is contained within a list of a_strings.
 *
 * @param needle the string to find
 * @param haystacks the strings that the string might be
 * @param len the number of strings in the haystack
 */
bool as_in(const a_string* restrict needle, const a_string** haystack, usize len);

/**
 * checks if a target string is contained within a list of C strings.
 *
 * @param needle the string to find
 * @param haystacks the strings that the string might be
 * @param len the number of strings in the haystack
 */
bool as_in_cstr(const a_string* restrict needle, const char** haystack, usize len);

/**
 * converts an a_string to a double.
 *
 * this is basically a wrapper around `strtod`. check strtod(3) for info
 * regarding the return value on overflow or underflow, and supported inputs.
 *
 * @param src the source string to convert
 * @param res pointer to the resultant number, which will be set on success.
 * @return index of first invalid character. if it is equal to the length of the
 * string, the conversion is successful. returns ERANGE on a range error.
 */
usize as_to_double(const a_string* restrict src, double* res);

/**
 * converts an a_string to a int64_t.
 *
 * this is basically a wrapper around `strtoll`. check strtol(3) for info
 * regarding the return value on overflow/underflow, and supported inputs.
 *
 * @param src the source
 * @param res pointer to the resultant number, which will be set on success.
 * @param base base of the number. check strtol(3) for more info.
 * @return index of the first invalid character, if it is equal to the length of
 * the string, the conversion is successful. returns ERANGE on a range error.
 */
usize as_to_integer(const a_string* restrict src, int64_t* res, int base);

// === UNICODE (UTF-8) STUFF ===
// string of 32 bit unicode codepoints

AV_DECL(dchar, a_dstring);

/**
 * counts how many UTF-8 codepoints are in an a_string
 *
 * @param s the string
 * @return the number of UTF-8 characters there are
 */
usize au_len(const a_string* restrict s);

/**
 * checks if a UTF-8 slice is valid.
 * @param s the string
 */
bool au_slice_valid(const u8* s, usize len);

/**
 * checks if a UTF-8 a_string is valid.
 *
 * @param s the string
 */
bool au_valid(const a_string* restrict s);

/**
 * (STATIC) decode a single unicode codepoint from the ptr to
 * the leading byte. Assumes and asserts that the char is valid.
 *
 * @param ptr the pointer to the leading byte
 * @return the char
 */
dchar au_decode(u8* ptr);

/**
 * (STATIC) encode a single UTF-32 codepoint into UTF-8.
 *
 * @param dest the destination array, 4 bytes long
 * @param src the UTF-32 dchar
 * @return the number of bytes used in the final encoding/
 */
u8 au_encode_cp(u8 dest[4], dchar src);

/**
 * returns the address of the nth UTF-8 codepoint in an a_string.
 *
 * @param s the string
 * @param idx the index
 * @return the address of the beginning of the codepoint
 */
u8* au_pos(const a_string* restrict s, usize idx);

#define au_iter(s, vname) for (u8* vname = au_next_begin((s), NULL); vname; vname = au_next_begin((s), vname))

/**
 * returns the next position of a UTF-8 codepoint given a beginning memory
 * location within the bounds of the a_string buffer.
 *
 * This is useful for manually iterating over an a_string efficiently.
 *
 * @param s the string
 * @param ptr the address, can be left NULL to assume the beginning of the
 * string. Returns NULL on error/stop.
 */
u8* au_next_begin(const a_string* restrict s, u8* begin);

/**
 * returns the next unicode character given a beginning memory location within
 * the bounds of the a_string buffer.
 *
 * This is useful for manually iterating over an a_string efficiently.
 *
 * @param s the string
 * @param ptr the address, can be left NULL to assume the beginning of the
 * string.
 * @return the char
 */
dchar au_next_codepoint(const a_string* restrict s, u8* begin);

/**
 * returns the unicode character at the nth index of a UTF-8 a_string.
 *
 * @param s the string
 * @param idx the index
 * @return the decoded codepoint
 */
dchar au_at(const a_string* restrict s, usize idx);

/**
 * returns an a_vector (heap dynamic array) of UTF-32 codepoints.
 *
 * @param s the string
 * @return the new UTF-32 string (must be freed).
 */
a_dstring au_codepoints(const a_string* restrict s);

void au_append_char(a_string* restrict s, dchar cp);

// asserts slice is valid
void au_append_slice(a_string* restrict s, const u8* data, usize len);

#endif // A_STRING_H
