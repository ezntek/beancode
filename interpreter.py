from parser import *

@dataclass
class Variable:
    val: BCValue
    const: bool

    def is_uninitialized(self) -> bool:
        return self.val.is_uninitialized()

class Intepreter:
    block: list[Statement]
    variables: dict[str, Variable]

    def __init__(self, block: list[Statement]) -> None:
        self.block = block
        self.variables = dict()

    @classmethod
    def new(cls, block: list[Statement]) -> "Interpreter": # type: ignore
        return cls(block)

    def visit_binaryexpr(self, expr: BinaryExpr) -> BCValue: # type: ignore
        match expr.op:
            case "assign":
                raise ValueError("impossible to have assign in binaryexpr")
            case "equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                res = (lhs == rhs)
                return BCValue("boolean", boolean=res)
            case "not_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                res = (lhs != rhs)
                return BCValue("boolean", boolean=res)
            case "greater_than":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                lhs_num: int | float
                rhs_num: int | float

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real # type: ignore
               
                    if rhs.kind not in ["integer", "real"]:
                        panic(f"impossible to perform greater_than between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num > rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        panic(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        panic(f"illegal to compare booleans")
                    elif lhs.kind == "string":
                        return BCValue("boolean", boolean=(lhs.get_string() > rhs.get_string()))
            case "less_than":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)

                lhs_num: int | float
                rhs_num: int | float

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real # type: ignore
               
                    if rhs.kind not in ["integer", "real"]:
                        panic(f"impossible to perform less_than between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        panic(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        panic(f"illegal to compare booleans")
                    elif lhs.kind == "string":
                        return BCValue("boolean", boolean=(lhs.get_string() < rhs.get_string()))
            case "greater_than_or_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                lhs_num: int | float
                rhs_num: int | float

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real # type: ignore
               
                    if rhs.kind not in ["integer", "real"]:
                        panic(f"impossible to perform greater_than_or_equal between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num >= rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        panic(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        panic(f"illegal to compare booleans")
                    elif lhs.kind == "string":
                        return BCValue("boolean", boolean=(lhs.get_string() >= rhs.get_string()))
            case "less_than_or_equal":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                lhs_num: int | float
                rhs_num: int | float

                if lhs.kind in ["integer", "real"]:
                    lhs_num = lhs.integer if lhs.integer is not None else lhs.real # type: ignore
               
                    if rhs.kind not in ["integer", "real"]:
                        panic(f"impossible to perform less_than_or_equal between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        panic(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        panic(f"illegal to compare booleans")
                    elif lhs.kind == "string":
                        return BCValue("boolean", boolean=(lhs.get_string() <= rhs.get_string()))
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

    def visit_expr(self, expr: Expr) -> BCValue: #type: ignore
        if isinstance(expr, Grouping):
            return self.visit_expr(expr.inner)
        elif isinstance(expr, Identifier):
            var = self.variables[expr.ident]
            if var.val == None or var.is_uninitialized():
                panic("attempted to access an uninitialized variable")
            return var.val
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

    def visit_if_stmt(self, stmt: IfStatement):
        cond: BCValue = self.visit_expr(stmt.cond)

        if cond.boolean:
            blk = self.new(stmt.if_block)
        else:
            blk = self.new(stmt.else_block)
        
        blk.variables = self.variables
        blk.visit_block(None)
            

    def visit_while_stmt(self, stmt: WhileStatement):
        cond: Expr = stmt.cond # type: ignore

        block: list[Statement] = stmt.while_s.block # type: ignore

        loop_intp = self.new(block)
        loop_intp.variables = self.variables # scope

        while self.visit_expr(cond).boolean:
            loop_intp.visit_block(block)

    def visit_for_stmt(self, stmt: ForStatement):
        begin = stmt.begin
        end = stmt.end
        step = stmt.step if stmt.step is not None else 1
       
        block = self.new(stmt.block)
        block.variables = self.variables

        counter = BCValue("integer", integer=begin)
        block.variables[stmt.counter.ident] = Variable(counter, const=False)

        if step > 0:
            while counter.get_integer() <= end:
                block.visit_block(None)
                counter.integer = counter.integer + step # type: ignore
        elif step < 0:
            while counter.get_integer() >= end:
                block.visit_block(None)
                counter.integer = counter.integer + step # type: ignore

    def visit_stmt(self, stmt: Statement):
        match stmt.kind:
            case "if":
                self.visit_if_stmt(stmt.if_s) # type: ignore
            case "for":
                self.visit_for_stmt(stmt.for_s) # type: ignore
            case "while":
                self.visit_while_stmt(stmt.while_s) # type: ignore
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

    def visit_block(self, block: list[Statement] | None):
        if block is not None:
            for stmt in block:
                self.visit_stmt(stmt)
        elif self.block is not None:
            for stmt in self.block:
                self.visit_stmt(stmt)

    def visit_program(self, program: Program):
        if program is not None:
            self.visit_block(program.stmts)
