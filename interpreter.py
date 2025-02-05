from parser import *

@dataclass
class Variable:
    val: BCValue
    const: bool

class Intepreter:
    ast: Program
    variables: dict[str, Variable]

    def __init__(self, ast: Program) -> None:
        self.ast = ast
        self.variables = dict()

    def visit_binaryexpr(self, expr: BinaryExpr) -> BCValue: # type: ignore
        match expr.op:
            case "assign":
                raise ValueError("impossible to have assign in binaryexpr")
            # add sub mul div
            case "mul":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                if lhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot multiply between bools, chars, and strings!")

                if rhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot multiply between bools, chars, and strings!")

                lhs_num: int | float = 0
                rhs_num : int | float = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()
                
                res = lhs_num * rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "div":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                if lhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot divide between bools, chars, and strings!")

                if rhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot divide between bools, chars, and strings!")

                lhs_num: int | float = 0
                rhs_num : int | float = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()
                
                res = lhs_num / rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "add":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                if lhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot add bools, chars, and strings!")

                if rhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot add bools, chars, and strings!")

                lhs_num: int | float = 0
                rhs_num : int | float = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()
                
                res = lhs_num + rhs_num
                
                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)
            case "sub":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                if lhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot subtract bools, chars, and strings!")

                if rhs.kind in ["boolean", "char", "string"]:
                    raise ValueError("Cannot subtract bools, chars, and strings!")

                lhs_num: int | float = 0
                rhs_num : int | float = 0

                if lhs.kind == "integer":
                    lhs_num = lhs.get_integer()
                elif lhs.kind == "real":
                    lhs_num = lhs.get_real()

                if rhs.kind == "integer":
                    rhs_num = rhs.get_integer()
                elif lhs.kind == "real":
                    rhs_num = rhs.get_real()
                
                res = lhs_num - rhs_num

                if isinstance(res, int):
                    return BCValue("integer", integer=res)
                elif isinstance(res, float):
                    return BCValue("real", real=res)

            case _:
                raise NotImplementedError("not implemented")

    def visit_expr(self, expr: Expr) -> BCValue: #type: ignore
        if isinstance(expr, Grouping):
            return self.visit_expr(expr.inner)
        elif isinstance(expr, Identifier):
            return self.variables[expr.ident].val
        elif isinstance(expr, Literal):
            return expr.to_bcvalue()
        elif isinstance(expr, BinaryExpr):
            return self.visit_binaryexpr(expr)
        else:
            raise ValueError("expr is very corrupted whoops")

    def visit_output_stmt(self, stmt: OutputStatement):
        res = ""
        for item in stmt.items:
            evaled = self.visit_expr(item)
            res += str(evaled) # TODO: proper displaying
        print(res)

    def visit_stmt(self, stmt: Statement):
        match stmt.kind:
            case "if":
                raise NotImplementedError("if not implemented")
            case "for":
                raise NotImplementedError("for not implemented")
            case "while":
                raise NotImplementedError("while not implemented")
            case "output":
                self.visit_output_stmt(stmt.output) # type: ignore
            case "assign":
                s: AssignStatement = stmt.assign # type: ignore
                key = s.ident.ident

                if key not in self.variables:
                    panic(f"attempted to use variable {key} without declaring it")

                if self.variables[key].const:
                    panic(f"attemped to write to constant {key}")

                self.variables[key].val = self.visit_expr(s.value)
            case "constant":
                c: ConstantStatement = stmt.constant # type: ignore
                key = c.ident.ident

                if key in self.variables:
                    panic(f"variable {key} declared!")

                self.variables[key] = Variable(c.value.to_bcvalue(), True)
            case "declare":
                d: DeclareStatement = stmt.declare # type: ignore
                key = d.ident.ident
                
                if key in self.variables:
                    panic(f"variable {key} declared!")

                self.variables[key] = Variable(BCValue(kind=d.typ), False)

    def visit_program(self):
        for stmt in self.ast.stmts:
            self.visit_stmt(stmt)
