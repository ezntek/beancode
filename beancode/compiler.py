# beancode: a portable IGCSE Computer Science (0478, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
#

import ast

from beancode.bean_ast import *


class Compiler:
    block: list[Statement]

    def __init__(self, block: list[Statement]) -> None:
        self.block = block

    def visit_array_literal(self, expr: ArrayLiteral):
        pass

    def visit_binaryexpr(self, expr: BinaryExpr) -> ast.expr:
        # XXX: we must asume no type errors, as there is no static way to detect for runtime type errors
        # (i.e. input/assign-declaring a variable and then performing ops on them). The optimizer must
        # have already checked it anyway, let us trust the programmer

        COMPARE_TABLE = {
            Operator.EQUAL: ast.Eq(),
            Operator.NOT_EQUAL: ast.NotEq(),
            Operator.GREATER_THAN: ast.Gt(),
            Operator.LESS_THAN: ast.Lt(),
            Operator.GREATER_THAN_OR_EQUAL: ast.GtE(),
            Operator.LESS_THAN_OR_EQUAL: ast.LtE(),
        }
        BINOP_TABLE = {
            Operator.POW: ast.Pow(),
            Operator.MUL: ast.Mult(),
            Operator.DIV: ast.Div(),
            Operator.ADD: ast.Add(),
            Operator.SUB: ast.Sub(),
            Operator.FLOOR_DIV: ast.FloorDiv(),
            Operator.MOD: ast.Mod(),
            Operator.AND: ast.And(),
            Operator.OR: ast.Or(),
        }

        match expr.op:
            case Operator.ASSIGN:
                raise RuntimeError("impossible to have assign in binaryexpr!")
            case (
                Operator.EQUAL
                | Operator.NOT_EQUAL
                | Operator.GREATER_THAN
                | Operator.LESS_THAN
                | Operator.GREATER_THAN_OR_EQUAL
                | Operator.LESS_THAN_OR_EQUAL
            ):
                return ast.Compare(
                    ops=[COMPARE_TABLE[expr.op]],
                    left=self.visit_expr(expr.lhs),
                    comparators=[self.visit_expr(expr.rhs)],
                )
            case (
                Operator.POW
                | Operator.MUL
                | Operator.DIV
                | Operator.ADD
                | Operator.SUB
                | Operator.FLOOR_DIV
                | Operator.MOD
                | Operator.AND
                | Operator.OR
            ):
                return ast.BinOp(
                    op=BINOP_TABLE[expr.op],
                    left=self.visit_expr(expr.lhs),
                    right=self.visit_expr(expr.rhs),
                )

    def visit_array_index(self, expr: ArrayIndex) -> ast.expr:
        value = None
        if expr.idx_inner:
            value = ast.Subscript(value=self.visit_expr(expr.expr), slice=self.visit_expr(expr.idx_inner), ctx=ast.Load()) 
        else:
            value = self.visit_expr(expr.expr)
        return ast.Subscript(value, slice=self.visit_expr(expr.idx_outer), ctx=ast.Load())

    def visit_fncall(self, expr: FunctionCall):
        pass

    def visit_literal(self, expr: Literal) -> ast.expr:
        if expr.val.kind == BCPrimitiveType.BOOLEAN:
            return ast.Constant(value=bool(expr.val.val))
        elif expr.val.is_array:
            raise ValueError("impossible to have an array here!")
        elif expr.val.kind_is_numeric():
            return ast.Constant(value=expr.val.val)  # type: ignore
        elif expr.val.kind_is_alpha():
            return ast.Constant(value=str(expr.val.val))

        return ast.Constant(value=None)

    def visit_typecast(self, tc: Typecast) -> ast.expr:
        raise NotImplementedError()

    def visit_expr(self, expr: Expr) -> ast.expr:
        match expr:
            case Typecast():
                return self.visit_typecast(expr)
            case Grouping():
                return self.visit_expr(expr.inner)
            case Negation():
                return ast.UnaryOp(op=ast.USub(), operand=self.visit_expr(expr.inner))
            case Not():
                return ast.UnaryOp(op=ast.Not(), operand=self.visit_expr(expr.inner))
            case Identifier():
                return ast.Name(id=expr.ident, ctx=ast.Load())
            case Literal():
                return self.visit_literal(expr)
            case ArrayLiteral():
                raise NotImplementedError("array literals not implemented")
            case BinaryExpr():
                return self.visit_binaryexpr(expr)
            case ArrayIndex():
                return self.visit_array_index(expr)
            case FunctionCall():
                pass
            case Sqrt():
                pass

    def visit_lvalue(self, lv: Lvalue):
        if isinstance(lv, ArrayIndex):
            pass
        else:
            pass

    def visit_if_stmt(self, stmt: IfStatement):
        pass

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        pass

    def visit_for_stmt(self, stmt: ForStatement):
        pass

    def visit_while_stmt(self, stmt: WhileStatement):
        pass

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        pass

    def visit_output_stmt(self, stmt: OutputStatement):
        pass

    def visit_input_stmt(self, stmt: InputStatement):
        pass

    def visit_return_stmt(self, stmt: ReturnStatement):
        pass

    def visit_argument_list(self, args: list[FunctionArgument]):
        pass

    def visit_procedure(self, stmt: ProcedureStatement):
        pass

    def visit_function(self, stmt: FunctionStatement):
        pass

    def visit_scope_stmt(self, stmt: ScopeStatement):
        pass

    def visit_include_stmt(self, stmt: IncludeStatement):
        pass

    def visit_call(self, stmt: CallStatement):
        pass

    def visit_assign_stmt(self, stmt: AssignStatement):
        pass

    def visit_constant_stmt(self, stmt: ConstantStatement):
        pass

    def visit_declare_stmt(self, stmt: DeclareStatement):
        pass

    def visit_trace_stmt(self, stmt: TraceStatement):
        pass

    def visit_fileid(self, file_id: Expr | str):
        pass

    def visit_openfile_stmt(self, stmt: OpenfileStatement):
        pass

    def visit_readfile_stmt(self, stmt: ReadfileStatement):
        pass

    def visit_writefile_stmt(self, stmt: WritefileStatement):
        pass

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        pass

    def visit_stmt(self, stmt: Statement):
        match stmt:
            case IfStatement():
                self.visit_if_stmt(stmt)
            case CaseofStatement():
                self.visit_caseof_stmt(stmt)
            case ForStatement():
                self.visit_for_stmt(stmt)
            case WhileStatement():
                self.visit_while_stmt(stmt)
            case RepeatUntilStatement():
                self.visit_repeatuntil_stmt(stmt)
            case OutputStatement():
                self.visit_output_stmt(stmt)
            case InputStatement():
                self.visit_input_stmt(stmt)
            case ReturnStatement():
                self.visit_return_stmt(stmt)
            case ProcedureStatement():
                self.visit_procedure(stmt)
            case FunctionStatement():
                self.visit_function(stmt)
            case ScopeStatement():
                self.visit_scope_stmt(stmt)
            case IncludeStatement():
                self.visit_include_stmt(stmt)
            case CallStatement():
                self.visit_call(stmt)
            case AssignStatement():
                self.visit_assign_stmt(stmt)
            case ConstantStatement():
                self.visit_constant_stmt(stmt)
            case DeclareStatement():
                self.visit_declare_stmt(stmt)
            case TraceStatement():
                self.visit_trace_stmt(stmt)
            case OpenfileStatement():
                self.visit_openfile_stmt(stmt)
            case ReadfileStatement():
                self.visit_readfile_stmt(stmt)
            case WritefileStatement():
                self.visit_writefile_stmt(stmt)
            case ClosefileStatement():
                self.visit_closefile_stmt(stmt)
            case ExprStatement():
                self.visit_expr(stmt.inner)
            case NewlineStatement():
                pass
            case CommentStatement():
                pass

    def visit_block(self, block: list[Statement] | None = None) -> list[ast.stmt]:
        # NOTE: To reduce code size, ALL CODE MUST RUN THROUGH THE OPTIMIZER.
        # Type checks are performed there.
        return []
