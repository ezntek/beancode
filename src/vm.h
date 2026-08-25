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
#ifndef BC_VM_H
#define BC_VM_H

#include "vm_types.h"

BCVM bc_vm_new(BCVM_Instr *src, usize src_len, BCValue *imms, usize imms_len);

void bc_vm_exec(BCVM *vm);

void bc_vm_free(BCVM *vm);

#endif // BC_VM_H
