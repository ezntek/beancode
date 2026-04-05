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
#ifndef BC_VM_TYPES_H
#define BC_VM_TYPES_H

#include "a_string.h"
#include "a_string_slice.h"
#include "common.h"

typedef enum {
    BC_TYPE_NULL = 0,
    BC_TYPE_INTEGER,
    BC_TYPE_REAL,
    BC_TYPE_CHAR,
    BC_TYPE_BOOLEAN,
    BC_TYPE_STRING,
    BC_TYPE_ARRAY,
    BC_TYPE_FUNCTION,
} BCType;

struct BCFunction;

typedef struct BCValue {
    BCType t;
    union {
        i64 i; // INTEGERs
        u8 c;  // CHARs, BOOLEANs
        f64 r; // REALs

        char* s; // STRINGs

        // a - sizeof(size_t): (ndim) how many dimensions
        // a - 2*sizeof(size_t): len(ndim - 1)
        // a - 3*sizeof(size_t): len(ndim - 2)
        // ...
        // a - n*sizeof(size_t): len(1st dim)
        struct BCValue* a; // ARRAYs

        struct BCFunction* f;
    } v;
} BCValue;

#define BCVALUE_STRING_LENGTH(val)                                             \
    *(usize*)((char*)((val)->v.s) - sizeof(usize))

typedef enum {
    BC_INSTR_NOP = 0,
    // Push one value onto the stack.
    BC_INSTR_PUSH,    // imm
    BC_INSTR_POP,     // ()
    BC_INSTR_LOAD,    // var
    BC_INSTR_STORE,   // var
    BC_INSTR_JMP,     // addr
    BC_INSTR_JF,      // ()
    BC_INSTR_JT,      // ()
    BC_INSTR_OUTPUT,  // imm
    BC_INSTR_INPUT,   // ()
    BC_INSTR_NOT,     // ()
    BC_INSTR_NEG,     // ()
    BC_INSTR_CMP_LT,  // ()
    BC_INSTR_CMP_GT,  // ()
    BC_INSTR_CMP_LTE, // ()
    BC_INSTR_CMP_GTE, // ()
    BC_INSTR_CMP_EQ,  // ()
    BC_INSTR_AND,     // ()
    BC_INSTR_OR,      // ()
    BC_INSTR_ADD,     // ()
    BC_INSTR_SUB,     // ()
    BC_INSTR_MUL,     // ()
    BC_INSTR_DIV,     // ()
    BC_INSTR_POW,     // ()
} BCVM_Opcode;

typedef u32 BCVM_Instr;

AV_DECL(BCValue, BCVM__Vars)
AV_DECL(BCValue, BCVM__Stack)

typedef struct {
    BCVM__Vars vars;
    BCVM__Stack stack;
} BCVM_Frame;

typedef struct {
    //
} BCFunction;

#define BCVM_INSTR_OPCODE(ins) (BCVM_Opcode)((ins) >> 26)

// 00000011 11111111 11111111 11111111
// 0   3    f   f    f   f    f   f
#define BCVM_INSTR_OPERAND(ins) (u32)((ins) & 0x03ffffff)

// push "hello"
// output
// push '\n'
// output

#endif // BC_VM_TYPES_H
