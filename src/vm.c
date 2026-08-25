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
#include <stdlib.h>
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>

#include "common.h"
#include "vec.h"
#include "vm.h"
#include "vm_types.h"

#define OPCODE(ins) BCVM_INSTR_OPCODE(ins)
#define JMP_SRC(ins) BCVM_INSTR_JMP_SRC(ins)
#define VALUE(ins) BCVM_INSTR_VALUE(ins)
#define DEST(ins) BCVM_INSTR_DEST(ins)
#define FNID(ins) BCVM_INSTR_FNID(ins)
#define SRC1(ins) BCVM_INSTR_SRC1(ins)
#define SRC2(ins) BCVM_INSTR_SRC2(ins)

#define IP (vm->frame->ip)
#define FUNC (vm->frame->func)

BCVM bc_vm_new(BCVM_Instr *src, usize src_len, BCValue *imms, usize imms_len) {
    BCVM vm = {
        .src = src, .src_len = src_len, .stack = {0}, .imms = {0}, .vars = {0}};

    vec_reserve(&vm.stack, 32);
    vec_reserve(&vm.vars, 8);
    vec_append_many(&vm.imms, imms, imms_len);

    return vm;
}

void bc_vm_free(BCVM *vm) {
    vec_free(&vm->stack);
    vec_free(&vm->imms);
    vec_free(&vm->vars);
}

static void exec_nop(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    (void)(ins);
}

static void exec_load_imm(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 immid = VALUE(ins);
    frame->regs[dest] = vm->imms.data[immid];
}

static void exec_load_integer(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 sv = VALUE(ins);
    // Move the sign bit into the top bit of the i32, and on the right shift the
    // CPU sign extends for us!
    i32 res = (i32)(sv << 10) >> 10;
    frame->regs[dest] = (BCValue){
        .type = BC_TYPE_INTEGER,
        .v.i = res,
    };
}

static void exec_load_boolean(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 sv = VALUE(ins);
    frame->regs[dest] = (BCValue){
        .type = BC_TYPE_BOOLEAN,
        .v.c = sv >= 1,
    };
}

static void exec_load_char(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 sv = VALUE(ins);
    frame->regs[dest] = (BCValue){
        .type = BC_TYPE_CHAR,
        .v.c = (u8)sv,
    };
}

static void exec_load_var(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 slot = VALUE(ins);
    frame->regs[dest] = frame->stack.data[slot];
}

static void exec_store(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 slot = VALUE(ins);

    frame->stack.data[slot] = frame->regs[dest];
}

static void exec_new_var(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    BCType t = DEST(ins);
    u32 slot = VALUE(ins);

    assert(slot == frame->stack.len);

    BCValue val = {
        .type = t | BCTYPE_UNINITIALIZED_MASK,
    };
}

static void exec_copy(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src = SRC1(ins);
    frame->regs[dest] = frame->regs[src];
}

static void exec_jmp(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
}

static void exec_jmp_false(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
}

static void exec_jmp_true(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
    u32 addr = VALUE(ins);
    if (frame->regs[dest].v.c) {
        vm->frame->ip = addr;
    }
}

static void exec_cmp_gt(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);

    BCValue *l = &vm->frame->regs[src1];
    BCValue *r = &vm->frame->regs[src1];
}

static void exec_cmp_lt(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_cmp_gte(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_cmp_lte(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_cmp_eq(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_add(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_sub(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_mul(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_div(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_pow(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_call(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u16 fnid = FNID(ins);
    u8 retreg = SRC1(ins);
    u8 nargs = SRC2(ins);
}

static void exec_fficall(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_index(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_index_matrix_h(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_index_matrix_t(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins), src1 = SRC1(ins), src2 = SRC2(ins);
}

static void exec_ret(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins) {
    u8 dest = DEST(ins);
}

typedef void (*InstrHandler)(BCVM *vm, BCVM_Frame *frame, BCVM_Instr ins);
static const InstrHandler JUMP_TABLE[] = {
    [BC_INSTR_NOP] = exec_nop,
    [BC_INSTR_LOAD_IMM] = exec_load_imm,
    [BC_INSTR_LOAD_INTEGER] = exec_load_integer,
    [BC_INSTR_LOAD_BOOLEAN] = exec_load_boolean,
    [BC_INSTR_LOAD_CHAR] = exec_load_char,
    [BC_INSTR_LOAD_VAR] = exec_load_var,
    [BC_INSTR_STORE] = exec_store,
    [BC_INSTR_NEW_VAR] = exec_new_var,
    [BC_INSTR_COPY] = exec_copy,
    [BC_INSTR_JMP] = exec_jmp,
    [BC_INSTR_JMP_FALSE] = exec_jmp_false,
    [BC_INSTR_JMP_TRUE] = exec_jmp_true,
    [BC_INSTR_CMP_GT] = exec_cmp_gt,
    [BC_INSTR_CMP_LT] = exec_cmp_lt,
    [BC_INSTR_CMP_GTE] = exec_cmp_gte,
    [BC_INSTR_CMP_LTE] = exec_cmp_lte,
    [BC_INSTR_CMP_EQ] = exec_cmp_eq,
    [BC_INSTR_ADD] = exec_add,
    [BC_INSTR_SUB] = exec_sub,
    [BC_INSTR_MUL] = exec_mul,
    [BC_INSTR_DIV] = exec_div,
    [BC_INSTR_POW] = exec_pow,
    [BC_INSTR_CALL] = exec_call,
    [BC_INSTR_FFICALL] = exec_fficall,
    [BC_INSTR_INDEX] = exec_index,
    [BC_INSTR_INDEX_MATRIX_H] = exec_index_matrix_h,
    [BC_INSTR_INDEX_MATRIX_T] = exec_index_matrix_t,
    [BC_INSTR_RET] = exec_ret,
};

void bc_vm_exec(BCVM *vm) {
    for (IP = 0; IP < FUNC->instrs_len; IP++) {
        BCVM_Instr ins = FUNC->instrs[IP];
        JUMP_TABLE[OPCODE(ins)](vm, vm->frame, ins);
    }
}
