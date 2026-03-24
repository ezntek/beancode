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

#include <cctype>
#include <ranges>
#include <string_view>

#include "common.hpp"

namespace beancode::util {

bool case_consistent(const std::string_view s) {
    if (!s.length()) return true;

    bool upper = isupper(s[0]);
    for (const char c : s) {
        if (isupper(c) != upper) return false;
    }

    return true;
}

bool case_compare(const std::string_view l, const std::string_view r) {
    if (l.length() != r.length()) return false;

    for (usize i = 0; i < l.length(); i++)
        if (toupper(l[i]) != toupper(r[i])) return false;

    return true;
}

}; // namespace beancode::util
