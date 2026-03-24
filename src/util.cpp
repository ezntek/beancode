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
#include <string_view>

#include "common.hpp"

namespace beancode::util {

bool case_consistent(const std::string_view s) {
    if (!s.length()) return true;

    int upper = isupper(s[0]);
    for (usize i = 1; i < s.length(); i++) {
        if (static_cast<int>(isupper(s[i])) != upper) return false;
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
