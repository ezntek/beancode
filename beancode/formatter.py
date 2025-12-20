from beancode.libroutines import LIBROUTINES
from .bean_ast import *


def _reverse_escape_code(ch: str) -> str:
    match ch:
        case "\n":
            return "\\n"
        case "\r":
            return "\\r"
        case "\033":
            return "\\e"
        case "\a":
            return "\\a"
        case "\b":
            return "\\b"
        case "\f":
            return "\\f"
        case "\v":
            return "\\v"
        case "\0":
            return "\\0"
        case "\\":
            return "\\\\"
        case _:
            return ch


class Formatter:
    block: list[Statement]
    buf: list[str]  # our string builder
    indent: int  # number of spaces to indent by
    end: int
    skip_newline: bool

    def __init__(self, block: list[Statement], indent=4):
        self.indent = indent
        self.buf = []
        self.block = block
        self.end = 0
        self.skip_newline = False

    def write(self, s: str):
        self.buf.append(s)
        self.end += 1

    def visit_type(self, typ: Type):
        if isinstance(typ, ArrayType):
            self.write("ARRAY[")
            bounds = typ.bounds
            self.visit_expr(bounds[0])
            self.write(":")
            self.visit_expr(bounds[1])
            if typ.is_matrix():
                self.write(",")
                self.visit_expr(bounds[2])  # type: ignore
                self.write(":")
                self.visit_expr(bounds[3])  # type: ignore
            self.write(f"] OF {typ.inner}")
        else:
            self.write(str(typ).upper())

    def visit_array_literal(self, expr: ArrayLiteral):
        self.write("{")
        for i, itm in enumerate(expr.items):
            self.visit_expr(itm)
            if i != len(expr.items) - 1:
                self.write(", ")
        self.write("}")

    def visit_binaryexpr(self, expr: BinaryExpr):
        if expr.op in {Operator.FLOOR_DIV, Operator.MOD}:
            self.write(expr.op.as_symbol())
            self.write("(")
            self.visit_expr(expr.lhs)
            self.write(", ")
            self.visit_expr(expr.rhs)
            self.write(")")
        else:
            self.visit_expr(expr.lhs)
            self.write(" ")
            self.write(expr.op.as_symbol())
            self.write(" ")
            self.visit_expr(expr.rhs)

    def visit_array_index(self, expr: ArrayIndex):
        self.visit_expr(expr.expr)
        self.write("[")
        self.visit_expr(expr.idx_outer)
        if expr.idx_inner:
            self.write(",")
            self.visit_expr(expr.idx_inner)
        self.write("]")

    def visit_fncall(self, expr: FunctionCall):
        if is_case_consistent(expr.ident) and expr.ident.lower() in LIBROUTINES:
            self.write(expr.ident.upper())
        else:
            self.write(expr.ident)
        self.write("(")
        for i, itm in enumerate(expr.args):
            self.visit_expr(itm)
            if i != len(expr.args) - 1:
                self.write(", ")
        self.write(")")

    def visit_literal(self, expr: Literal):
        match expr.val.kind:
            case BCPrimitiveType.STRING:
                new = "".join([_reverse_escape_code(c) for c in expr.val.get_string()])
                self.write(f'"{new}"')
            case BCPrimitiveType.CHAR:
                self.write(f"'{_reverse_escape_code(expr.val.get_char())}'")
            case BCPrimitiveType.NULL:
                self.write(f"NULL")
            case _:
                self.write(str(expr.val))

    def visit_expr(self, expr: Expr):
        match expr:
            case Typecast():
                self.visit_type(expr.typ)
                self.write("(")
                self.visit_expr(expr.expr)
                self.write(")")
            case Grouping():
                self.write("(")
                self.visit_expr(expr.inner)
                self.write(")")
            case Negation():
                self.write("-")
                self.visit_expr(expr.inner)
            case Not():
                self.write("NOT ")
                self.visit_expr(expr.inner)
            case Identifier():
                self.write(expr.ident)
            case Literal():
                self.visit_literal(expr)
            case ArrayLiteral():
                return self.visit_array_literal(expr)
            case BinaryExpr():
                return self.visit_binaryexpr(expr)
            case ArrayIndex():
                return self.visit_array_index(expr)
            case FunctionCall():
                return self.visit_fncall(expr)
            case Sqrt():
                self.write("SQRT(")
                self.visit_expr(expr.inner)
                self.write(")")

    def visit_lvalue(self, lv: Lvalue):
        if isinstance(lv, ArrayIndex):
            self.visit_array_index(lv)
        else:
            self.write(str(lv.ident))

    def reduce_from(self, idx: int):
        orig_len = len(self.buf)
        s = "".join(self.buf[idx:])
        for _ in range(orig_len - idx):
            self.buf.pop()
        self.buf.append(s)
        self.end = idx + 1

    def write_block(self, block: list[Statement], extra=0):
        for line in self.visit_block(block):
            if line.isspace():
                self.write("\n")
            else:
                self.write(" " * (self.indent + extra) + line)

    def visit_if_stmt(self, stmt: IfStatement):
        saved_end = self.end
        self.write("IF ")
        self.visit_expr(stmt.cond)
        self.write("\n")
        self.reduce_from(saved_end)  # create one line
        s = " " * (self.indent // 2)
        self.write(f"{s}THEN\n")
        self.write_block(stmt.if_block)
        if stmt.else_block:
            self.write(f"{s}ELSE\n")
            self.write_block(stmt.else_block)
        self.write("ENDIF\n")

    def visit_caseof_stmt(self, stmt: CaseofStatement):
        saved_end = self.end
        self.write("CASE OF ")
        self.visit_expr(stmt.expr)
        self.write("\n")
        self.reduce_from(saved_end)
        # XXX: magic f---ery to get the indentation right
        saved_cur = self.end
        saved_buf = self.buf
        new_buf: list[str] = []
        self.buf = new_buf
        for i, branch in enumerate(stmt.branches):
            self.end = i
            saved_end = self.end
            if isinstance(branch, NewlineStatement):
                if self.skip_newline:
                    self.skip_newline = False
                    self.write("")
                else:
                    self.write("\n")
                continue

            if isinstance(branch, Comment):
                self.visit_comment(branch)
                if self.skip_newline:
                    self.skip_newline = False
                else:
                    self.write("\n")
                continue

            self.visit_expr(branch.expr)
            self.write(": ")
            s = branch.stmt
            if isinstance(s, IfStatement):
                self.write("IF ")
                self.visit_expr(s.cond)
                self.write(" THEN\n")
                self.reduce_from(saved_end)
                self.write_block(s.if_block)
                if s.else_block:
                    self.write("ELSE\n")
                    self.write_block(s.else_block)
                self.write("ENDIF\n")
            else:
                self.visit_stmt(s, raw=True)
                self.write("\n")
                self.reduce_from(saved_end)
        if stmt.otherwise:
            saved_end = self.end
            self.write("OTHERWISE ")
            self.visit_stmt(stmt.otherwise, raw=True)
            self.write("\n")
            self.reduce_from(saved_end)
        self.buf = saved_buf
        self.end = saved_cur
        for line in new_buf:
            if line.isspace():
                self.write("\n")
            else:
                self.write(" " * self.indent + line)
        self.write("ENDCASE\n")

    def visit_for_stmt(self, stmt: ForStatement):
        saved_end = self.end
        self.write(f"FOR {stmt.counter.ident} <- ")
        self.visit_expr(stmt.begin)
        self.write(" TO ")
        self.visit_expr(stmt.end)
        if stmt.step:
            self.write("STEP")
            self.visit_expr(stmt.step)
        self.write("\n")
        self.reduce_from(saved_end)
        self.write_block(stmt.block)
        self.write(f"NEXT {stmt.counter.ident}\n")

    def visit_while_stmt(self, stmt: WhileStatement):
        saved_end = self.end
        self.write("WHILE ")
        self.visit_expr(stmt.cond)
        self.write(" DO\n")
        self.reduce_from(saved_end)
        self.reduce_from(saved_end)
        self.write_block(stmt.block)
        self.write("ENDWHILE\n")

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        self.write("REPEAT\n")
        self.write_block(stmt.block)
        saved_end = self.end
        self.write("UNTIL ")
        self.visit_expr(stmt.cond)
        self.write("\n")
        self.reduce_from(saved_end)

    def visit_output_stmt(self, stmt: OutputStatement):
        if stmt.newline:
            self.write("OUTPUT ")
        else:
            self.write("PRINT ")

        if not stmt.items:
            self.write('""')
            return
        for i, itm in enumerate(stmt.items):
            self.visit_expr(itm)
            if i != len(stmt.items) - 1:
                self.write(", ")

    def visit_input_stmt(self, stmt: InputStatement):
        self.write("INPUT ")
        self.visit_lvalue(stmt.ident)

    def visit_return_stmt(self, stmt: ReturnStatement):
        self.write("RETURN")
        if stmt.expr:
            self.write(" ")
            self.visit_expr(stmt.expr)

    def visit_argument_list(self, args: list[FunctionArgument]):
        if not args:
            return
        self.write("(")
        for i, arg in enumerate(args):
            self.write(f"{arg.name}: ")
            self.visit_type(arg.typ)
            if i != len(args) - 1:
                self.write(", ")
        self.write(")")

    def visit_procedure(self, stmt: ProcedureStatement):
        saved_end = self.end
        self.write(f"PROCEDURE {stmt.name}")
        self.visit_argument_list(stmt.args)
        self.write("\n")
        self.reduce_from(saved_end)
        self.write_block(stmt.block)
        self.write("ENDPROCEDURE\n")

    def visit_function(self, stmt: FunctionStatement):
        saved_end = self.end
        self.write(f"FUNCTION {stmt.name}")
        self.visit_argument_list(stmt.args)
        self.write(" RETURNS ")
        self.visit_type(stmt.returns)
        self.write("\n")
        self.reduce_from(saved_end)
        self.write_block(stmt.block)
        self.write("ENDFUNCTION\n")

    def visit_scope_stmt(self, stmt: ScopeStatement):
        self.write("SCOPE\n")
        self.write_block(stmt.block)
        self.write("ENDSCOPE\n")

    def visit_include_stmt(self, stmt: IncludeStatement):
        if stmt.ffi:
            self.write("INCLUDE_FFI ")
        else:
            self.write("INCLUDE ")
        self.write(f'"{stmt.file}"')

    def visit_call(self, stmt: CallStatement):
        self.write(f"CALL {stmt.ident}")
        if stmt.args:
            self.write("(")
            for i, itm in enumerate(stmt.args):
                self.visit_expr(itm)
                if i != len(stmt.args) - 1:
                    self.write(", ")
            self.write(")")

    def visit_assign_stmt(self, stmt: AssignStatement):
        self.visit_lvalue(stmt.ident)
        self.write(" <- ")
        self.visit_expr(stmt.value)

    def visit_constant_stmt(self, stmt: ConstantStatement):
        if stmt.export:
            self.write("EXPORT ")
        self.write("CONSTANT ")
        self.visit_lvalue(stmt.ident)
        self.write(" <- ")
        self.visit_expr(stmt.value)

    def visit_declare_stmt(self, stmt: DeclareStatement):
        self.write("DECLARE ")
        for i, itm in enumerate(stmt.ident):
            self.write(itm.ident)
            if i != len(stmt.ident) - 1:
                self.write(", ")
        self.write(": ")
        self.visit_type(stmt.typ)

    def visit_trace_stmt(self, stmt: TraceStatement):
        saved_end = self.end
        self.write("TRACE")
        if stmt.vars:
            self.write("(")
            for i, var in enumerate(stmt.vars):
                self.write(var)
                if i != len(stmt.vars) - 1:
                    self.write(", ")
            self.write(")")
        if stmt.file_name:
            self.write(f' TO "{stmt.file_name}"')
        self.write("\n")
        self.reduce_from(saved_end)
        self.write_block(stmt.block)
        self.write("ENDTRACE\n")

    def visit_fileid(self, file_id: Expr | str):
        if isinstance(file_id, Grouping):
            self.write("(")
            self.visit_expr(file_id.inner)
            self.write(")")
        elif isinstance(file_id, Expr):
            self.write(" ")
            self.visit_expr(file_id)
        else:
            self.write(f" {file_id}")

    def visit_openfile_stmt(self, stmt: OpenfileStatement):
        self.write("OPENFILE")
        self.visit_fileid(stmt.file_ident)
        self.write(" FOR ")

        seen = False
        if stmt.mode[0]:
            if not seen:
                seen = True
            else:
                self.write(" AND ")
            self.write("READ")
        if stmt.mode[1]:
            if not seen:
                seen = True
            else:
                self.write(" AND ")
            self.write("WRITE")
        if stmt.mode[2]:
            if not seen:
                seen = True
            else:
                self.write(" AND ")
            self.write("APPEND")

    def visit_readfile_stmt(self, stmt: ReadfileStatement):
        self.write("READFILE ")
        self.visit_fileid(stmt.file_ident)
        self.write(", ")
        self.visit_lvalue(stmt.target)

    def visit_writefile_stmt(self, stmt: WritefileStatement):
        self.write("WRITEFILE ")
        self.visit_fileid(stmt.file_ident)
        self.write(", ")
        self.visit_expr(stmt.src)

    def visit_closefile_stmt(self, stmt: ClosefileStatement):
        self.write("CLOSEFILE")
        self.visit_fileid(stmt.file_ident)

    def visit_comment(self, com: Comment):
        if not com.multiline:
            buf = "//"
            if com.data:
                s = com.data[0]
                if s and s[0] != " ":
                    buf += " "
                buf += s
            self.write(buf)
            self.skip_newline = True
            return
        buf = ""
        for i, line in enumerate(com.data):
            new = line
            if i == 0:
                buf += "/*"
            elif line.lstrip()[:1] != "*":
                buf += " *"
            else:
                i = line.rfind("*")
                new = line[i + 1 :]
                buf += " *"
            if line:
                if line[0] != " ":
                    buf += " "
                buf += new
            self.write(buf + "\n")
            buf = ""
        self.write(" */")

    def visit_stmt(self, stmt: Statement, next=None, raw=False) -> bool:
        if not raw:
            saved_end = self.end
        match stmt:
            case IfStatement():
                self.visit_if_stmt(stmt)
                return False
            case CaseofStatement():
                self.visit_caseof_stmt(stmt)
                return False
            case ForStatement():
                self.visit_for_stmt(stmt)
                return False
            case WhileStatement():
                self.visit_while_stmt(stmt)
                return False
            case RepeatUntilStatement():
                self.visit_repeatuntil_stmt(stmt)
                return False
            case OutputStatement():
                self.visit_output_stmt(stmt)
            case InputStatement():
                self.visit_input_stmt(stmt)
            case ReturnStatement():
                self.visit_return_stmt(stmt)
            case ProcedureStatement():
                self.visit_procedure(stmt)
                return False
            case FunctionStatement():
                self.visit_function(stmt)
                return False
            case ScopeStatement():
                self.visit_scope_stmt(stmt)
                return False
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
                return False
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
                if self.skip_newline:
                    self.skip_newline = False
                else:
                    self.write("\n")
                return False
            case CommentStatement():
                self.visit_comment(stmt.comment)
        if not raw:
            rv = False

            if (
                isinstance(next, CommentStatement)
                and not next.comment.multiline
                and next.pos.row == stmt.pos.row
            ):
                self.write(" ")
                self.visit_comment(next.comment)
                rv = True

            self.write("\n")
            self.reduce_from(saved_end)  # type: ignore
            return rv

        return False

    def visit_block(self, block: list[Statement] | None = None) -> list[str]:
        # XXX: heavy abuse of Python object pointer semantics here.
        # We create a new list every visit_block, and replace the pointer to the
        # active buffer in the formatter before we visit the block. after the fact,
        # we replace the previous buffer and return the new one ;)
        saved_end = self.end
        saved_buf = self.buf  # pointer copy, to not lose it forever
        # create the new array
        new_buf = []
        self.buf = new_buf  # spoof-o-matic
        blk = block if block != None else self.block
        skip = False
        self.end = 0
        for i, stmt in enumerate(blk):
            if skip:
                skip = False
                continue
            if isinstance(stmt, NewlineStatement) and i == len(blk) - 1:
                continue
            res = self.visit_stmt(stmt, next=blk[i + 1] if i != len(blk) - 1 else None)
            if res and not skip:
                skip = True
        # pretend that nothing happened
        self.buf = saved_buf
        self.end = saved_end
        return new_buf
