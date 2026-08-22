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

#include "vec.h"
#include "common.h"
#include "vm.h"
#include "vm_types.h"

#define OPCODE(ins)  BCVM_INSTR_OPCODE(ins)
#define OPERAND(ins) BCVM_INSTR_OPERAND(ins)

#define TOP (vm->stack.len)

BCVM bc_vm_new(BCVM_Instr* src, usize src_len, BCValue* imms, usize imms_len) {
    BCVM vm = {
        .src = src, .src_len = src_len, .stack = {0}, .imms = {0}, .vars = {0}};

    vec_reserve(&vm.stack, 32);
    vec_reserve(&vm.vars, 8);
    vec_append_many(&vm.imms, imms, imms_len);

    return vm;
}

void bc_vm_free(BCVM* vm) {
    vec_free(&vm->stack);
    vec_free(&vm->imms);
    vec_free(&vm->vars);
}

static void push(BCVM* vm, const BCVM_Instr ins);
static void pop(BCVM* vm);

static void output(BCVM* vm, const BCVM_Instr ins);

static void push(BCVM* vm, const BCVM_Instr ins) {
    BCValue* imm = &vm->imms.data[OPERAND(ins)];
    switch (imm->t) {
        case BC_TYPE_NULL:
        case BC_TYPE_INTEGER:
        case BC_TYPE_REAL:
        case BC_TYPE_BOOLEAN:
        case BC_TYPE_CHAR: {
            vec_append(&vm->stack, *imm);
        } break;
        case BC_TYPE_STRING: {
            // 1 usize for length
            usize len = BCVALUE_STRING_LENGTH(imm);
            char* resdata = malloc(sizeof(usize) + len + 1);
            check_alloc(resdata);
            *(usize*)resdata = len;
            resdata += sizeof(usize);
            strcpy(resdata, imm->v.s);
            BCValue val = {.t = BC_TYPE_STRING, .v.s = resdata};
            vec_append(&vm->stack, val);
        } break;
        case BC_TYPE_ARRAY: {
            panic("not implemented");
        } break;
    }
}

static void pop(BCVM* vm) {
    if (!vm->stack.len) panic("no items to pop off stack");

    BCValue* last = &vec_pop(&vm->stack);
    switch (last->t) {
        case BC_TYPE_NULL:
        case BC_TYPE_INTEGER:
        case BC_TYPE_REAL:
        case BC_TYPE_BOOLEAN:
        case BC_TYPE_CHAR: {
        } break;
        case BC_TYPE_STRING: {
            free(last->v.s - 8);
        } break;
        case BC_TYPE_ARRAY: {
            free(last->v.a);
        } break;
    }
}

static void output(BCVM* vm, const BCVM_Instr ins) {
    usize count = vm->imms.data[OPERAND(ins)].v.i;
    BCValue* top;
    for (; count; count--) {
        top = &vec_last(&vm->stack);

        // TODO: refactor
        switch (top->t) {
            case BC_TYPE_NULL: {
                printf("(null)");
            } break;
            case BC_TYPE_INTEGER: {
                printf("%ld", top->v.i);
            } break;
            case BC_TYPE_REAL: {
                printf("%lf", top->v.r);
            } break;
            case BC_TYPE_BOOLEAN: {
                if (top->v.c)
                    printf("TRUE");
                else
                    printf("FALSE");
            } break;
            case BC_TYPE_CHAR: {
                printf("%c", top->v.c);
            } break;
            case BC_TYPE_STRING: {
                usize len = BCVALUE_STRING_LENGTH(top);
                printf("%.*s", (int)len, top->v.s);
            } break;
            case BC_TYPE_ARRAY: {
                panic("arrays not implemented");
            } break;
        }

        pop(vm);
    }
}

void bc_vm_exec(BCVM* vm) {
    for (vm->cur = 0; vm->cur < vm->src_len; vm->cur++) {
        BCVM_Instr ins = vm->src[vm->cur];
        switch (OPCODE(ins)) {
            case BC_INSTR_NOP: continue;
            case BC_INSTR_PUSH: {
                push(vm, ins);
            } break;
            case BC_INSTR_POP: {
                pop(vm);
            } break;
            case BC_INSTR_LOAD: {
                panic("not implemented");
            } break;
            case BC_INSTR_STORE: {
                panic("not implemented");
            } break;
            case BC_INSTR_JMP: {
                panic("not implemented");
            } break;
            case BC_INSTR_JF: {
                panic("not implemented");
            } break;
            case BC_INSTR_JT: {
                panic("not implemented");
            } break;
            case BC_INSTR_OUTPUT: {
                output(vm, ins);
            } break;
            case BC_INSTR_INPUT: {
                panic("not implemented");
            } break;
            case BC_INSTR_NOT: {
                panic("not implemented");
            } break;
            case BC_INSTR_NEG: {
                panic("not implemented");
            } break;
            case BC_INSTR_CMP_LT: {
                panic("not implemented");
            } break;
            case BC_INSTR_CMP_GT: {
                panic("not implemented");
            } break;
            case BC_INSTR_CMP_LTE: {
                panic("not implemented");
            } break;
            case BC_INSTR_CMP_GTE: {
                panic("not implemented");
            } break;
            case BC_INSTR_CMP_EQ: {
                panic("not implemented");
            } break;
            case BC_INSTR_AND: {
                panic("not implemented");
            } break;
            case BC_INSTR_OR: {
                panic("not implemented");
            } break;
            case BC_INSTR_ADD: {
                panic("not implemented");
            } break;
            case BC_INSTR_SUB: {
                panic("not implemented");
            } break;
            case BC_INSTR_MUL: {
                panic("not implemented");
            } break;
            case BC_INSTR_DIV: {
                panic("not implemented");
            } break;
            case BC_INSTR_POW: {
                panic("not implemented");
            } break;
        }
    }
}
