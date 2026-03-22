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

#include "utf8.hpp"
#include <cctype>
#include <cstring>

namespace beancode::utf8 {

constexpr bool iscont(char c) {
    return ((uint8_t)c & 0xC0) == 0x80;
}

size_t len(const char* s, size_t buflen) {
    size_t len = 0;
    for (size_t i = 0; i < buflen; i++) {
        if (!iscont(s[i]))
            len++;
    }
    return len;
}

bool valid(const char* s, size_t len) {
    if (!s)
        return false;

    size_t cur_codepoint = 0, rembytes = 0, saved_rembytes = 0, i = 0;
    uint8_t ch = 0;
    for (i = 0; i < len; i++) {
        ch = s[i];
        if (iscont(ch)) {
            if (!rembytes)
                return false; // stray continuation

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

        if (saved_rembytes)
            return false;

        if (i + rembytes >= len)
            return false;

        if (rembytes) {
            uint8_t next = s[i + 1];
            if ((ch & 0xE0) == 0xC0) {
                // reject overlong, lead byte must be >=0b1100010
                if (ch < 0xC2)
                    return false;
            } else if ((ch & 0xF0) == 0xE0) {
                // overlong
                if (ch == 0xE0 && next < 0xA0)
                    return false;
                // reject surrogates
                if (ch == 0xED && next >= 0xA0)
                    return false;
            } else if ((ch & 0xF8) == 0xF0) {
                // overlong
                if (ch == 0xF0 && next < 0x90)
                    return false;
                // range
                if (ch == 0xF4 && next > 0x8F)
                    return false;
                // cannot encode >U+10FFFF
                if (ch > 0xF4)
                    return false;
            }
        }
    }

    if (rembytes)
        return false;

    return true;
}

// INFO: returns -1 on error
const char* codepoint_pos(const char* s, size_t len, size_t idx) {
    size_t cur_codepoint = 0;

    for (size_t i = 0; i < len; i++) {
        if (!iscont(s[i])) {
            if (cur_codepoint == idx)
                return s + i;
            cur_codepoint++;
        }
    }

    return nullptr;
}

// INFO: returns null on error
const char* next_codepoint_begin(const char* cur, const char* end) {
    if (!cur || cur >= end)
        return nullptr;

    unsigned char c = (unsigned char)*cur;
    if (c < 0x80)
        return cur + 1;

    // preincrement ensures we skip the leader
    while (++cur < end && iscont(*cur))
        continue;

    return (cur < end) ? cur : nullptr;
}

// INFO: returns -1 on error
int32_t decode(const char* ptr) {
    if (iscont(*ptr))
        return -1;

    int32_t res = 0;
    uint8_t initial = ptr[0];
    if ((initial & 0x80) == 0) {
        res = initial;
    } else if ((initial & 0xE0) == 0xC0) {
        res = (int32_t)((initial & 0x1F) << 6) | (ptr[1] & 0x3F);
    } else if ((initial & 0xF0) == 0xE0) {
        res = (int32_t)((initial & 0x0F) << 12) | (ptr[1] & 0x3F) << 6 | (ptr[2] & 0x3F);
        return ((res < 0x800)                       // overlong
                || (res >= 0xD800 && res <= 0xDFFF) // utf-16
                )
                   ? -1
                   : res;
    } else if ((initial & 0xF8) == 0xF0) {
        res = (int32_t)((initial & 0x07) << 18) | (ptr[1] & 0x3F) << 12 | (ptr[2] & 0x3F) << 6 | (ptr[3] & 0x3F);
        if (res < 0x10000 || res > 0x10FFFF)
            return -1; // overlong or out of unicode range
    } else {
        return -1;
    }

    return res;
}

// INFO: returns -1 on error
int32_t next_codepoint(const char* begin, const char* end) {
    const char* res = next_codepoint_begin(begin, end);
    if (!res)
        return -1;

    return decode(res);
}

// INFO: returns 0 on error
uint8_t encode_codepoint(char dest[4], int32_t src) {
    if (src > 0x10FFFF || (0xD800 <= src && src <= 0xDFFF)) {
        return 0;
    }

    memset((void*)dest, 0, 4);
    size_t len = 1 + (src > 0x7F) + (src > 0x7FF) + (src > 0xFFFF);
    // write continuation bytes in reverse
    for (size_t i = len - 1; i > 0; i--) {
        dest[i] = 0x80 | (src & 0x3F);
        src >>= 6;
    }

    static const uint8_t mask[5] = {0x00, 0x00, 0xC0, 0xE0, 0xF0};
    dest[0] = mask[len] | (uint8_t)src;
    return len;
}

// WARN: does not do any heap allocation
// INFO: returns new length
size_t append_char(char* s, size_t len, int32_t cp) {
    char buf[4] = {0};
    uint8_t count = encode_codepoint(buf, cp);
    strncat(s + len, buf, count);
    return len + count;
}

bool is_space(const char* s) {
    if (isspace(*s))
        return true;

    // U+2000 to U+200A
    int32_t res;
    if ((res = decode(s)) < 0)
        return false; // continuation/failure

    switch (res) {
        case 0x1680:
        case 0x2000:
        case 0x2001:
        case 0x2002:
        case 0x2003:
        case 0x2004:
        case 0x2005:
        case 0x2006:
        case 0x2007:
        case 0x2008:
        case 0x2009:
        case 0x200A:
        case 0x202F:
        case 0x205F:
        case 0x2028:
        case 0x2029:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

bool is_newline(const char* s) {
    if (*s == '\n')
        return true;

    int32_t res;
    if ((res = decode(s)) < 0)
        return false;

    switch (res) {
        case 0x2028:
        case 0x2029:
            return true;
        default:
            return false;
    }
}

} // namespace beancode::utf8
