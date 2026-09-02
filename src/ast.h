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
#ifndef BCAST_AST_H
#define BCAST_AST_H

#include "common.h"
#include "util.h"
#include "vec.h"
#include "vm_types.h"

typedef enum {
    BCAST_PRIM_INTEGER = 0,
    BCAST_PRIM_REAL = 0x01,
    BCAST_PRIM_FLOAT = 0x02,
    BCAST_PRIM_BOOLEAN = 0x03,
    BCAST_PRIM_CHAR = 0x04,
    BCAST_PRIM_STRING = 0x05,
    BCAST_PRIM_I64 = 0x06,
    BCAST_PRIM_I32 = 0x07,
    BCAST_PRIM_I16 = 0x08,
    BCAST_PRIM_I8 = 0x09,
    BCAST_PRIM_U64 = 0x0a,
    BCAST_PRIM_U32 = 0x0b,
    BCAST_PRIM_U16 = 0x0c,
    BCAST_PRIM_U8 = 0x0d,
    BCAST_PRIM_F32 = 0x0e,
    BCAST_PRIM_F64 = 0x0f,
} BCASTPrimitiveType;

typedef enum {
    BCAST_LITERAL_PRIMITIVE = 0,
    // NOTE: not implemented
    BCAST_LITERAL_ARRAY = 1,
} BCASTLiteralKind;

typedef struct {
    BCASTLiteralKind kind;
    union {
        BCValue value;
    };
} BCASTLiteral;

typedef enum {
    BCAST_EXPR_LITERAL = 0x01,
    BCAST_EXPR_IDENT = 0x02,
    BCAST_EXPR_FNCALL = 0x03,
    BCAST_EXPR_TYPECAST = 0x04,
    // unaries
    // 0001xxxx
    BCAST_EXPR_UNOT = 0x10,
    BCAST_EXPR_UNEG = 0x11,
    BCAST_EXPR_UBITNOT = 0x12,
    BCAST_EXPR_UREF = 0x13,
    BCAST_EXPR_UDEREF = 0x14,
    // binaries
    // 001xxxxx
    BCAST_EXPR_BINADD = 0x21,
    BCAST_EXPR_BINSUB = 0x22,
    BCAST_EXPR_BINMUL = 0x23,
    BCAST_EXPR_BINDIV = 0x24,
    BCAST_EXPR_BINPOW = 0x25,
    BCAST_EXPR_BINBITAND = 0x26,
    BCAST_EXPR_BINBITOR = 0x27,
    BCAST_EXPR_BINBITXOR = 0x28,
    BCAST_EXPR_BINDOT = 0x29,
    BCAST_EXPR_BINLT = 0x2a,
    BCAST_EXPR_BINGT = 0x2b,
    BCAST_EXPR_BINLEQ = 0x2c,
    BCAST_EXPR_BINGEQ = 0x2d,
    BCAST_EXPR_BINEQ = 0x2e,
    BCAST_EXPR_BINNEQ = 0x2f,
    BCAST_EXPR_BINSHL = 0x30,
    BCAST_EXPR_BINSHR = 0x31,
    BCAST_EXPR_BINOR = 0x32,
    BCAST_EXPR_BINAND = 0x33,
} BCASTExprKind;

typedef struct {
    BCASTExprKind kind;
    union {
        // for unaries
        u32 operand_id;
        // for binaries
        u32 lhs_id;
    };
    union {
        // for binaries
        u32 rhs_id;
        // for typecast
        BCASTPrimitiveType type;
    };
} BCASTExpr;

typedef enum {
    BCAST_STATEMENT_EXPR = 0,
} BCASTStatementKind;

typedef struct {
    BCASTStatementKind kind;
    u32 id; // discriminated according to StatementKind
} BCASTStatement;

typedef struct {
    BCASTStatement *stmts;
    usize stmts_len;
} BCASTProgram;

VEC_DECL(BCASTExpr, BCASTExprArray);

typedef struct {
    StringArray strings;
    BCASTExprArray exprs;
} BCASTStorage;

#endif
