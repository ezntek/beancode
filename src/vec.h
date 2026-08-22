/*

  pkgit - package it!

  Copyright (C) 2026 dacctal
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef PKGIT_VEC_H
#define PKGIT_VEC_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#define VEC_INITIAL_SIZE 5

#define VEC_DECL(T, name)                                                      \
	typedef struct {                                                           \
		T *data;                                                               \
		size_t len;                                                            \
		size_t cap;                                                            \
	} name

#define vec_append(v, item)                                                    \
	do {                                                                       \
		if ((v)->len + 1 > (v)->cap) {                                         \
			if ((v)->cap == 0)                                                 \
				(v)->cap = VEC_INITIAL_SIZE;                                   \
			else                                                               \
				(v)->cap += ((v)->cap >> 1);                                   \
			(v)->data = realloc((v)->data, sizeof(*(v)->data) * (v)->cap);     \
			check_alloc((v)->data);                                            \
		}                                                                      \
		(v)->data[(v)->len++] = (item);                                        \
	} while (0)

#define vec_clear(v)                                                           \
	do {                                                                       \
		memset((v)->data, 0, sizeof(*(v)->data) * (v)->cap);                   \
		(v)->len = 0;                                                          \
	} while (0)

#define vec_free(v)                                                            \
	do {                                                                       \
		if ((v)->data) {                                                       \
			free((v)->data);                                                   \
			(v)->data = NULL;                                                  \
		}                                                                      \
	} while (0)

#define vec_reserve(v, amt)                                                    \
	do {                                                                       \
		if (amt >= (v)->len) {                                                 \
			(v)->data = realloc((v)->data, sizeof(*(v)->data) * amt);          \
			check_alloc((v)->data);                                            \
			(v)->cap = amt;                                                    \
		}                                                                      \
	} while (0)

#define vec_append_many(v, itms, itms_len)                                     \
	do {                                                                       \
		if ((v)->len + itms_len > (v)->cap) {                                  \
			(v)->cap += itms_len;                                              \
			(v)->data = realloc((v)->data, sizeof(*(v)->data) * (v)->cap);     \
			check_alloc((v)->data);                                            \
		}                                                                      \
		memcpy(&(v)->data[(v)->len], itms, sizeof(*(v)->data) * itms_len);     \
		(v)->len += itms_len;                                                  \
	} while (0)

#define vec_last(v) ((v)->data[(assert((v)->len > 0), (v)->len - 1)])

#define vec_at(v, pos)                                                         \
	((v)->data[(assert(0 <= (pos) && (pos) < (v)->len), pos)])

#define vec_pop(v) ((v)->data[(assert((v)->len > 0), --(v)->len)])

#define vec_pop_many(v, count)                                                 \
	do {                                                                       \
		assert(count < (v)->len);                                              \
		(v)->len -= count;                                                     \
	} while (0)

#endif // PKGIT_VEC_H
