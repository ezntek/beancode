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
#pragma once

#include "common.hpp"

namespace beancode::utf8 {

constexpr bool iscont(char c);

size_t len(const char* s, size_t buflen);

bool valid(const char* s, size_t len);

// Gets a pointer to the beginning of the next codepoint.
// NOTE: null on error
const char* codepoint_pos(const char* s, size_t len, size_t idx);

// Gets a pointer to the beginning of the next codepoint
// NOTE: nullptr on error
const char* next_codepoint_begin(const char* cur, const char* end);

// Decode utf-8 into utf-32
// NOTE: -1 on error
int32_t decode(const char* ptr);

// Gets the next valid codepoint from begin
// NOTE: -1 on error
int32_t next_codepoint(const char* begin, const char* end);

// Encodes a utf-32 codepoint into utf-8
// NOTE: 0 on error
uint8_t encode_codepoint(char dest[4], int32_t src);

// Appends a utf-32 codepoint to a byte buffer
// len: current length of byte buffer
// WARN: it does not do any heap allocation
// INFO: returns new length
size_t append_char(char* s, size_t len, int32_t cp);

// Is it a unicode space character?
bool is_space(const char* s);

// Is it a unicode newline character?
bool is_newline(const char* s);

} // namespace beancode::utf8
