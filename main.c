/*
 * beancode: a very cursed IGCSE pseudocode interpreter.
 *
 * Copyright (c) Eason Qin, 2025.
 *
 * This source code form is licensed under the Mozilla Public License version
 * 2.0. You may find the full text of the license in the root of the project, or
 * visit the OSI website for a digital version.
 *
 */

#include <stdio.h>

#include "3rdparty/asv/a_string.h"

int main(void) {
    a_string s = astr("and the pain begins...");
    printf("%s\n", s.data);
    a_string_free(&s);
}
