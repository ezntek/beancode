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

from beancode import humanize_index
from beancode.bean_ast import *
from beancode.bean_ffi import BCParamSpec

class TypeEntry:
    __slots__ = ('typ', 'const')

    typ: BCType
    const: bool

    def __init__(self, t: BCType, const: bool):
        self.typ = t
        self.const = const

class Compiler:
    block: list[Statement]
    vars: list[dict[str, TypeEntry]]
    # list of a table between function name and args (which is a table between the arg name and type)
    funcs: list[dict[str, dict[str, BCType]]]

    def __init__(self, block: list[Statement]) -> None:
        self.block = block
        self.vars = []
        self.funcs = []

    # XXX: all the get_<node>_type functions throw exceptions i.e. report errors
    # because I'm too lazy
    def get_ident_type(self, expr: Identifier) -> BCType:
        for itm in reversed(self.vars):
            if expr.ident in itm:
                return itm[expr.ident].typ

        raise BCError(f'cannot access undeclared variable "{expr.ident}"', expr.pos)

    def get_binaryexpr_type(self, expr: BinaryExpr) -> BCType:
        lhs = self.get_expr_type(expr.lhs)
        rhs = self.get_expr_type(expr.rhs)

        match expr.op:
            case Operator.EQUAL | Operator.NOT_EQUAL:
                return BCPrimitiveType.BOOLEAN
            case (
                Operator.LESS_THAN
                | Operator.LESS_THAN_OR_EQUAL
                | Operator.GREATER_THAN
                | Operator.GREATER_THAN_OR_EQUAL
            ):
                if not is_type_numeric(lhs):
                    raise BCError(
                        f"cannot have {lhs} in left hand side of {expr.op.humanize()}!",
                        expr.lhs.pos,
                    )

                if not is_type_numeric(rhs):
                    raise BCError(
                        f"cannot have {rhs} in right hand side of {expr.op.humanize()}!",
                        expr.rhs.pos,
                    )

                return BCPrimitiveType.BOOLEAN
            case Operator.AND | Operator.OR | Operator.NOT:
                if lhs != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot have {lhs} in left hand side of {expr.op.humanize()}!",
                        expr.lhs.pos,
                    )

                if rhs != BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot have {rhs} in right hand side of {expr.op.humanize()}!",
                        expr.rhs.pos,
                    )

                return BCPrimitiveType.BOOLEAN
            case Operator.ADD:
                if lhs == BCPrimitiveType.STRING or rhs == BCPrimitiveType.STRING:
                    return BCPrimitiveType.STRING

                if lhs == BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot have BOOLEAN in left hand side of addition!",
                        expr.lhs.pos,
                    )

                if rhs == BCPrimitiveType.BOOLEAN:
                    raise BCError(
                        f"cannot have BOOLEAN in right hand side of addition!",
                        expr.rhs.pos,
                    )

                if lhs == BCPrimitiveType.REAL or rhs == BCPrimitiveType.REAL:
                    return BCPrimitiveType.REAL
                else:
                    return BCPrimitiveType.INTEGER
            case _:
                if expr.op not in {Operator.FLOOR_DIV, Operator.MOD} and not (
                    is_type_numeric(lhs) and is_type_numeric(rhs)
                ):
                    # everything else
                    raise BCError(
                        f"cannot {expr.op.humanize()} between {lhs} and {rhs}", expr.pos
                    )

        raise RuntimeError("unreachable")

    def get_expr_type(self, expr: Expr) -> BCType:
        match expr:
            case Typecast():
                # TODO: do type checks
                return expr.typ
            case Grouping():
                return self.get_expr_type(expr.inner)
            case Negation():
                t = self.get_expr_type(expr.inner)
                if t != BCPrimitiveType.INTEGER and t != BCPrimitiveType.REAL:
                    raise BCError(f"cannot negate a value of type {t}!", expr.pos)
                return t
            case Not():
                t = self.get_expr_type(expr.inner)
                if t != BCPrimitiveType.BOOLEAN:
                    raise BCError(f"cannot NOT a value of type {t}!", expr.pos)
                return t
            case Identifier():
                return self.get_ident_type(expr)
            case Literal():
                return expr.val.kind
            case ArrayLiteral():
                raise NotImplementedError("array literals will be added later")
            case BinaryExpr():
                return self.get_binaryexpr_type(expr)
            case ArrayIndex():
                raise NotImplementedError("array indices not supported")
            case FunctionCall():
                raise NotImplementedError("function calls not supported")
            case Sqrt():
                t = self.get_expr_type(expr.inner)
                if t != BCPrimitiveType.INTEGER and t != BCPrimitiveType.REAL:
                    raise BCError(
                        f"cannot get the square root of a value of type {t}!", expr.pos
                    )
                return t
        raise RuntimeError("unreachable")

    def visit_type(self, typ: Type, pos: Pos) -> BCType:
        if isinstance(typ, ArrayType):
            if not typ.bounds:
                raise BCError("Cannot have unbounded arrays in compiled beancode", pos)

            new_bounds = []
            for i, itm in enumerate(typ.bounds):
                if not isinstance(itm, Literal):
                    raise BCError(f"{humanize_index(i+1)} bound to array must be a constant value known at compile-time!", itm.pos)
                v = itm.val
                if v.kind != BCPrimitiveType.INTEGER:
                    raise BCError(f"{humanize_index(i+1)} bound to array must be an INTEGER, not {v.kind}!")
                new_bounds.append(v.get_integer()) 
            return BCArrayType(bounds=tuple(new_bounds), inner=typ.inner)
        else:
            return typ # type: ignore

    def visit_ident(self, expr: Identifier) -> ast.expr:
        for itm in reversed(self.vars):
            if expr.ident in itm:
                return ast.Name(id=expr.ident, ctx=ast.Load())

        raise BCError(f'cannot access undeclared variable "{expr.ident}"')

    def visit_array_literal(self, expr: ArrayLiteral):
        raise NotImplementedError("array literals not implemented")

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

        raise RuntimeError("Unreachable")

    def visit_array_index(self, expr: ArrayIndex) -> ast.expr:
        value = None
        if expr.idx_inner:
            value = ast.Subscript(
                value=self.visit_expr(expr.expr),
                slice=self.visit_expr(expr.idx_inner),
                ctx=ast.Load(),
            )
        else:
            value = self.visit_expr(expr.expr)
        return ast.Subscript(
            value, slice=self.visit_expr(expr.idx_outer), ctx=ast.Load()
        )

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

    def visit_function_call(self, expr: FunctionCall) -> ast.expr:
        raise NotImplementedError()

    def visit_sqrt(self, expr: Sqrt) -> ast.expr:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="math", ctx=ast.Load()), attr="sqrt", ctx=ast.Load()
            ),
            args=[self.visit_expr(expr.inner)],
            keywords=[],
        )

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
                return self.visit_ident(expr)
            case Literal():
                return self.visit_literal(expr)
            case ArrayLiteral():
                raise NotImplementedError("array literals not implemented")
            case BinaryExpr():
                return self.visit_binaryexpr(expr)
            case ArrayIndex():
                return self.visit_array_index(expr)
            case FunctionCall():
                return self.visit_function_call(expr)
            case Sqrt():
                return self.visit_sqrt(expr)
        raise RuntimeError("unreachable")

    def visit_lvalue(self, lv: Lvalue) -> ast.expr:
        if isinstance(lv, ArrayIndex):
            raise RuntimeError("lvalue array indexes not implemented")
        else:
            _ = self.get_ident_type(lv)
            return ast.Name(id=lv.ident, ctx=ast.Store())

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

    def visit_output_stmt(self, stmt: OutputStatement) -> ast.stmt:
        args = []
        for arg in stmt.items:
            args.append(self.visit_expr(arg))
        return ast.Expr(
            ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=args, keywords=[])
        )

    def visit_input_stmt(self, stmt: InputStatement):
        target = self.visit_lvalue(stmt.ident)
        call = ast.Call(func=ast.Name(id="input", ctx=ast.Load()), args=[], keywords=[])

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
        if isinstance(stmt.ident, ArrayIndex):
            raise RuntimeError("not implemented")
        else:
            lhs = self.visit_lvalue(stmt.ident)
            rhs = self.visit_expr(stmt.value)
            return ast.Assign(targets=[lhs], value=rhs, lineno=0)

    def visit_constant_stmt(self, stmt: ConstantStatement):
        if isinstance(stmt.value, Literal):
            typ = stmt.value.val.kind 
        else:
            typ = self.get_expr_type(stmt.value)
        self.vars[-1][stmt.ident.ident] = TypeEntry(typ, True)
        exp = self.visit_expr(stmt.value)
        return ast.Assign(targets=[ast.Name(id=stmt.ident.ident, ctx=ast.Store())], value=exp, lineno=0)

    def visit_declare_stmt(self, stmt: DeclareStatement):
        typ = self.visit_type(stmt.typ, stmt.pos)
        for name in stmt.ident:
            self.vars[-1][name.ident] = TypeEntry(typ, False)

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

    def visit_stmt(self, stmt: Statement) -> ast.stmt | None:
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
                return self.visit_output_stmt(stmt)
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
                return self.visit_assign_stmt(stmt)
            case ConstantStatement():
                return self.visit_constant_stmt(stmt)
            case DeclareStatement():
                return self.visit_declare_stmt(stmt)
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
                exp = self.visit_expr(stmt.inner)
                return ast.Expr(value=exp)
            case NewlineStatement():
                raise RuntimeError("unreachable")
            case CommentStatement():
                raise RuntimeError("unreachable")
        raise NotImplementedError("aaaaaaa")

    def visit_block(self, block: list[Statement] | None = None) -> list[ast.stmt]:
        # NOTE: To reduce code size, ALL CODE MUST RUN THROUGH THE OPTIMIZER.
        # Type checks are performed there.
        blk = block if block is not None else self.block
        res = []
        self.vars.append(dict())
        self.funcs.append(dict())
        for stmt in blk:
            s = self.visit_stmt(stmt)
            if s:
                res.append(s)
        self.vars.pop()
        self.funcs.pop()
        return res

    def visit_program(self) -> ast.Module:
        return ast.Module(body=self.visit_block(), type_ignores=[])
