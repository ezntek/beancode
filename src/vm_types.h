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

#include "common.h"
#include "str.h"
#include "vec.h"

struct BCFunction;

typedef struct BCValue {
    // type >> 5 == 0: non-primitive type ID
    u64 type;
    union {
        i64 i; // INTEGERs
        u8 c;  // CHARs, BOOLEANs
        f64 r; // REALs

        char *s; // STRINGs

        // a - sizeof(size_t): (ndim) how many dimensions
        // a - 2*sizeof(size_t): len(ndim - 1)
        // a - 3*sizeof(size_t): len(ndim - 2)
        // ...
        // a - n*sizeof(size_t): len(1st dim)
        struct BCValue *a; // ARRAYs

        struct BCFunction *f;

        // TODO: implement structures, fields or oop
    } v;
} BCValue;

#define BCVALUE_STRING_LENGTH(val)                                             \
    *(usize *)((char *)((val)->v.s) - sizeof(usize))

// _H: head of two part instr
// _T: tail of two part instr
typedef enum {
    BC_INSTR_NOP = 0,
    BC_INSTR_LOAD_IMM,       // immid
    BC_INSTR_LOAD_INTEGER,   // dest=src, sv
    BC_INSTR_LOAD_BOOLEAN,   // dest=src, sv
    BC_INSTR_LOAD_CHAR,      // dest=src, sv
    BC_INSTR_LOAD_VAR,       // dest=src, slot
    BC_INSTR_STORE,          // dest=src, slot=dest
    BC_INSTR_NEW_VAR,        // dest=type, slot,
    BC_INSTR_COPY,           // dest, src1=src, src2=unused
    BC_INSTR_DEEP_COPY,      // dest, src1=src, src2=unused
    BC_INSTR_JMP,            // dest=unused, instr no (addr)
    BC_INSTR_JMP_FALSE,      // dest=val, instr no (addr)
    BC_INSTR_JMP_TRUE,       // dest=val, instr no (addr)
    BC_INSTR_CMP_GT,         // dest, src1, src2
    BC_INSTR_CMP_LT,         // dest, src1, src2
    BC_INSTR_CMP_GTE,        // dest, src1, src2
    BC_INSTR_CMP_LTE,        // dest, src1, src2
    BC_INSTR_CMP_EQ,         // dest, src1, src2
    BC_INSTR_ADD,            // dest, src1, src2
    BC_INSTR_SUB,            // dest, src1, src2
    BC_INSTR_MUL,            // dest, src1, src2
    BC_INSTR_DIV,            // dest, src1, src2
    BC_INSTR_POW,            // dest, src1, src2
    BC_INSTR_CALL,           // fnid, retreg, nargs
    BC_INSTR_FFICALL,        // dest=retreg, src1=namereg, src2=nargs
    BC_INSTR_INDEX,          // dest, src1=arr, src2=idx
    BC_INSTR_INDEX_MATRIX_H, // dest, src1=arr, src2=idx1
    BC_INSTR_INDEX_MATRIX_T, // dest=unused, src1=idx2, src2=unused
    BC_INSTR_RET,            // dest=unused, src1=reg, src2=unused
} BCVM_Opcode;

typedef u32 BCVM_Instr;
// NOTE: we only use the bottom 4 bits of 8 bit register values.
// | opcode (6) |  src (4) |------------- imm ID (22) -----------  |
// | opcode (6) |  src (4) |-------- small value / sv (22) ------  |
// | opcode (6) |  src (4) |------------- var ID (22) -----------  |
// | opcode (6) |  src (4) | ---------- instr no. (22) ----------  |
// | opcode (6) | ---- fnid (10) ----- |  retreg (8)   | nargs (8) |
// | opcode (6) |  (2)  |   dest (8)   |   src1 (8)    | src2 (8)  |

// no mask needed, just shift right
#define BCVM_INSTR_OPCODE_MASK 0x0
#define BCVM_INSTR_JMP_SRC_MASK 0x03c00000
#define BCVM_INSTR_VALUE_MASK 0x003fffff
#define BCVM_INSTR_DEST_MASK 0x000f0000
#define BCVM_INSTR_FNID_MASK 0x03ff0000
#define BCVM_INSTR_SRC1_MASK 0x00000f00
#define BCVM_INSTR_SRC2_MASK 0x0000000f

#define BCVM_INSTR_OPCODE_SHIFT 25
#define BCVM_INSTR_JMP_SRC_SHIFT 22
#define BCVM_INSTR_VALUE_SHIFT 0
#define BCVM_INSTR_DEST_SHIFT 16
#define BCVM_INSTR_FNID_SHIFT 16
#define BCVM_INSTR_SRC1_SHIFT 8
#define BCVM_INSTR_SRC2_SHIFT 0

#define BCVM_INSTR_OPCODE(ins)                                                 \
    (BCVM_Opcode)((u32)(ins) >> BCVM_INSTR_OPCODE_MASK)
#define BCVM_INSTR_JMP_SRC(ins)                                                \
    (u8)(((u32)(ins) & BCVM_INSTR_JMP_SRC_MASK) >> 22)
#define BCVM_INSTR_VALUE(ins) ((u32)(ins) & BCVM_INSTR_VALUE_MASK)
#define BCVM_INSTR_DEST(ins)                                                   \
    (u8)(((u32)(ins) & BCVM_INSTR_DEST_MASK) >> BCVM_INSTR_DEST_SHIFT)
#define BCVM_INSTR_FNID(ins)                                                   \
    (u16)(((u32)(ins) & BCVM_INSTR_FNID_MASK) >> BCVM_INSTR_FNID_SHIFT)
#define BCVM_INSTR_SRC1(ins)                                                   \
    (u8)(((u32)(ins) & BCVM_INSTR_SRC1_MASK) >> BCVM_INSTR_SRC1_SHIFT)
#define BCVM_INSTR_SRC2(ins) (u8)((u32)(ins) & BCVM_INSTR_SRC2_MASK)

#define BCVM_INSTR_SET(attr, ins, val)                                         \
    (ins = (ins & ~BCVM_INSTR_##attr##_MASK) |                                 \
           ((val << BCVM_INSTR_##attr##_SHIFT) & BCVM_INSTR_##attr##_MASK))

VEC_DECL(BCValue, BCValueArray);

typedef struct BCVM_Function {
    BCVM_Instr *instrs;
    usize instrs_len;
    // owned slice
    str_view name;
} BCVM_Function;

typedef struct BCVM_Frame {
    struct BCVM_Frame *prev;
    BCVM_Function *func;
    BCValueArray stack;
    BCValue regs[16];
} BCVM_Frame;

typedef struct {
    BCValueArray imms, globs;
    BCVM_Frame *frames;
    u32 max_frames, cur_frame, ip;
} BCVM;

#endif // BC_VM_TYPES_H
