from parser import *
import typing as t

@dataclass
class Variable:
    val: BCValue
    const: bool

    def is_uninitialized(self) -> bool:
        return self.val.is_uninitialized()

BlockType = t.Literal["if", "while", "for", "repeatuntil", "function", "procedure"]

class Interpreter:
    block: list[Statement]
    variables: dict[str, Variable]
    functions: dict[str, ProcedureStatement|FunctionStatement]
    calls: list[BlockType]
    func: bool
    proc: bool
    loop: bool
    toplevel: bool
    retval: BCValue | None = None
    _returned: bool

    def __init__(self, block: list[Statement], func=False, proc=False, loop=False) -> None:
        self.block = block
        self.calls = list()
        self.variables = dict()
        self.functions = dict()
        self.func = func
        self.proc = proc
        self.loop = loop
        self._returned = False

    @classmethod
    def new(cls, block: list[Statement], func=False, proc=False, loop=False) -> "Interpreter": # type: ignore
        return cls(block, func=func, proc=proc, loop=loop) # type: ignore

    def can_return(self) -> tuple[bool, bool]:
        proc = False
        func = False

        for item in reversed(self.calls):
            if item == "procedure":
                proc = True
                break
            elif item == "function":
                func = True
                break

        return (proc, func) 


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
                        raise BCRuntimeError(f"impossible to perform greater_than between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num > rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCRuntimeError(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        raise BCRuntimeError(f"illegal to compare booleans")
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
                        raise BCRuntimeError(f"impossible to perform less_than between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCRuntimeError(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        raise BCRuntimeError(f"illegal to compare booleans")
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
                        raise BCRuntimeError(f"impossible to perform greater_than_or_equal between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num >= rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCRuntimeError(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        raise BCRuntimeError(f"illegal to compare booleans")
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
                        raise BCRuntimeError(f"impossible to perform less_than_or_equal between {lhs.kind} and {rhs.kind}")

                    rhs_num = rhs.integer if rhs.integer is not None else lhs.real # type: ignore

                    return BCValue("boolean", boolean=(lhs_num < rhs_num))
                else:
                    if lhs.kind != rhs.kind:
                        raise BCRuntimeError(f"cannot compare incompatible types {lhs.kind} and {rhs.kind}")
                    elif lhs.kind == "boolean":
                        raise BCRuntimeError(f"illegal to compare booleans")
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
            case "and":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                if lhs.kind != "boolean":
                    raise BCRuntimeError(f"cannot perform logical AND on value with type {lhs.kind}")
            
                if rhs.kind != "boolean":
                    raise BCRuntimeError(f"cannot perform logical AND on value with type {lhs.kind}")
                
                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()
               
                res = lhs_b and rhs_b
                return BCValue("boolean", boolean=res)
            case "or":
                lhs = self.visit_expr(expr.lhs)
                rhs = self.visit_expr(expr.rhs)
                
                if lhs.kind != "boolean":
                    raise BCRuntimeError(f"cannot perform logical OR on value with type {lhs.kind}")
            
                if rhs.kind != "boolean":
                    raise BCRuntimeError(f"cannot perform logical OR on value with type {lhs.kind}")
                
                lhs_b = lhs.get_boolean()
                rhs_b = rhs.get_boolean()
               
                # python does: False or True = False....fuck you
    
                res = lhs_b or rhs_b

                return BCValue("boolean", boolean=res)

    def _get_array_index(self, ind: ArrayIndex) -> tuple[int, int | None]:
        index = self.visit_expr(ind.idx_outer).integer

        if index is None:
            raise BCRuntimeError("found None for array index")
        
        v = self.variables[ind.ident.ident].val
        
        if isinstance(v.kind, BCArrayType):
            a: BCArray = v.array # type: ignore
            
            if a.typ.is_matrix:
                if ind.idx_inner is None:
                    raise BCRuntimeError("expected 2 indices for matrix indexing")
                
                inner_index = self.visit_expr(ind.idx_inner).integer
                if inner_index is None:
                    raise BCRuntimeError("found None for inner array index")

                return (index, inner_index)
            else:
                return (index, None)
        else:
            raise BCRuntimeError(f"attemped to index {v.kind}")


    def visit_array_index(self, ind: ArrayIndex) -> BCValue: # type: ignore
        index = self.visit_expr(ind.idx_outer).integer

        if index is None:
            raise BCRuntimeError("found None for array index")
        
        v = self.variables[ind.ident.ident].val
         
        if isinstance(v.kind, BCArrayType):
            if v.array is None:
                return BCValue.empty("null")

            a: BCArray = v.array # type: ignore
            
            tup = self._get_array_index(ind)
            if a.typ.is_matrix:
                inner = tup[1]
                if inner is None:
                    raise BCRuntimeError("second index not present for matrix index")
                
                if tup[0] not in range(a.matrix_bounds[0], a.matrix_bounds[1]+1): # type: ignore
                    raise BCRuntimeError("attempted to access out of bounds array element")
                
                if tup[1] not in range(a.matrix_bounds[2], a.matrix_bounds[3]+1): # type: ignore
                    raise BCRuntimeError("attempted to access out of bounds array element")

                res = a.matrix[tup[0]-1][inner-1] # type: ignore
                
                if res.is_uninitialized():
                    return BCValue.empty("null")
                else:
                    return res
            else:
                if tup[0] not in range(a.flat_bounds[0], a.flat_bounds[1]+1): # type: ignore
                    raise BCRuntimeError("attempted to access out of bounds array element")
                
                res = a.flat[tup[0]-1] # type: ignore
                if res.is_uninitialized():
                    return BCValue.empty("null")
                else:
                    return res
        else:
            raise BCRuntimeError(f"attempted to index {v.kind}")
 
    def visit_call(self, stmt: CallStatement):
        proc = self.functions[stmt.ident]

        if isinstance(proc, FunctionStatement):
            raise BCRuntimeError("cannot run CALL on a function! please call the function without the CALL keyword instead.")

        intp = self.new(proc.block, proc=True)
        intp.calls = self.calls
        intp.calls.append("procedure")
        vars = self.variables
        
        if len(proc.args) != len(stmt.args):
            raise BCRuntimeError(f"procedure {proc.name} declares {len(proc.args)} variables but only found {len(stmt.args)} in procedure call")

        for argdef, argval in zip(proc.args, stmt.args):
            val = self.visit_expr(argval)
            vars[argdef.name] = Variable(val=val, const=False)

        intp.variables = vars
        intp.functions = self.functions

        intp.visit_block(proc.block)

    def visit_fncall(self, stmt: FunctionCall) -> BCValue:
        func = self.functions[stmt.ident]

        if isinstance(func, ProcedureStatement):
            raise BCRuntimeError("cannot call procedure without CALL!")

        intp = self.new(func.block, func=True)
        intp.calls = self.calls
        intp.calls.append("function")
        vars = self.variables
        
        if len(func.args) != len(stmt.args):
            raise BCRuntimeError(f"function {func.name} declares {len(func.args)} variables but only found {len(stmt.args)} in procedure call")

        for argdef, argval in zip(func.args, stmt.args):
            val = self.visit_expr(argval)
            vars[argdef.name] = Variable(val=val, const=False)

        intp.variables = vars
        intp.functions = self.functions

        intp.visit_block(func.block)
        if intp._returned is False:
            raise BCRuntimeError(f"function did not return a value!")

        if intp.retval is None:
            raise BCRuntimeError(f"function's return value is None!")
        else:
            return intp.retval # type: ignore
        

    def visit_expr(self, expr: Expr) -> BCValue: #type: ignore
        if isinstance(expr, Grouping):
            return self.visit_expr(expr.inner)
        elif isinstance(expr, Negation):
            inner = self.visit_expr(expr.inner)
            if inner.kind not in ["integer", "real"]:
                raise BCRuntimeError(f"attemped to negate a value of type {inner.kind}")

            if inner.kind == "integer":
                return BCValue("integer", integer=-inner.integer) # type: ignore
            elif inner.kind == "real":
                return BCValue("real", real=-inner.real) # type: ignore
        elif isinstance(expr, Not):
            inner = self.visit_expr(expr.inner)
            if inner.kind != "boolean":
                raise BCRuntimeError(f"attempted to perform logical NOT on value of type {inner.kind}")

            return BCValue("boolean", boolean=not inner.get_boolean())
        elif isinstance(expr, Identifier):
            var = self.variables[expr.ident]
            if var.val == None or var.is_uninitialized():
                raise BCRuntimeError("attempted to access an uninitialized variable")
            return var.val
        elif isinstance(expr, Literal):
            return expr.to_bcvalue()
        elif isinstance(expr, BinaryExpr):
            return self.visit_binaryexpr(expr)
        elif isinstance(expr, ArrayIndex):
            return self.visit_array_index(expr) 
        elif isinstance(expr, FunctionCall):
            return self.visit_fncall(expr)

        else:
            raise ValueError("expr is very corrupted whoops")

    def _display_array(self, arr: BCArray) -> str:
        if not arr.typ.is_matrix:
            res = "["
            flat: list[BCValue] = arr.flat # type: ignore
            for idx, item in enumerate(flat):
                if item.is_uninitialized():
                    res += "(uninitialized)"
                else:
                    res += str(item)

                if idx != len(flat) - 1:
                    res += ", "
            res += ']'

            return res
        else:
            matrix: list[list[BCValue]] = arr.matrix # type: ignore
            outer_res = "["
            for oidx, a in enumerate(matrix):
                res = "["
                for iidx, item in enumerate(a):
                    if item.is_uninitialized():
                        res += "(uninitialized)"
                    else:
                        res += str(item)

                    if iidx != len(a) - 1:
                        res += ", "
                res += "]"

                outer_res += res
                if oidx != len(matrix) - 1:
                    outer_res += ", "
            outer_res += "]"

            return outer_res 

    def visit_output_stmt(self, stmt: OutputStatement):
        res = ""
        for item in stmt.items:
            evaled = self.visit_expr(item)
            if isinstance(evaled.kind, BCArrayType):
                res += self._display_array(evaled.array) # type: ignore
            else:
                res += str(evaled)
        print(res)

    def visit_input_stmt(self, stmt: InputStatement):
        inp = input()
        id = stmt.ident.ident

        data: Variable | None = self.variables.get(id)

        if data is None:
            raise BCRuntimeError(f"attempted to call `INPUT` into nonexistent variable {id}")

        if data.const:
            raise BCRuntimeError(f"attempted to call `INPUT` into constant {id}")

        match data.val.kind:
            case "string":
                self.variables[id].val.kind = "string"
                self.variables[id].val.string = inp
            case "char":
                if len(inp) > 1:
                    raise BCRuntimeError(f"expected single character but got `{inp}` for CHAR")

                self.variables[id].val.kind = "char"
                self.variables[id].val.char = inp
            case "boolean":
                if inp.lower() not in ["true", "false", "yes", "no"]:
                    raise BCRuntimeError(f"expected TRUE, FALSE, YES or NO including lowercase for BOOLEAN but got `{inp}`")

                inp = inp.lower()
                if inp in ["true", "yes"]:
                    self.variables[id].val.kind = "boolean"
                    self.variables[id].val.boolean = True
                elif inp in ["false", "no"]:
                    self.variables[id].val.kind = "boolean"
                    self.variables[id].val.boolean = False
            case "integer":
                inp = inp.lower().strip()
                p = Parser([])
                if p.is_integer(inp):
                    try:
                        res = int(inp)
                        self.variables[id].val.kind = "integer"
                        self.variables[id].val.integer = res
                    except ValueError:
                        raise BCRuntimeError("expected INTEGER for INPUT")
                else:
                    raise BCRuntimeError("expected INTEGER for INPUT")
            case "real":
                inp = inp.lower().strip()
                p = Parser([])
                if p.is_real(inp) or p.is_integer(inp):
                    try:
                        res = float(inp)
                        self.variables[id].val.kind = "real"
                        self.variables[id].val.real = res
                    except ValueError:
                        raise BCRuntimeError("expected REAL for INPUT")
                else:
                    raise BCRuntimeError("expected REAL for INPUT")

    def visit_return_stmt(self, stmt: ReturnStatement): 
        proc, func = self.can_return()

        if not proc and not func:
            raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

        if func:
            if stmt.expr is None:
                raise BCRuntimeError("you must return something from a function!")

            res = self.visit_expr(stmt.expr)
            self.retval = res
            self._returned = True
        elif proc:
            if stmt.expr is not None:
                raise BCRuntimeError("you cannot return a value from a procedure!")
            
            self._returned = True

    def visit_if_stmt(self, stmt: IfStatement):
        cond: BCValue = self.visit_expr(stmt.cond)

        if cond.boolean:
            intp: Interpreter = self.new(stmt.if_block)
        else:
            intp: Interpreter = self.new(stmt.else_block)
        
        intp.variables = self.variables
        intp.functions = self.functions
        intp.calls = self.calls
        intp.calls.append("if") # type: ignore
        intp.visit_block(None)
        if intp._returned:
            proc, func = self.can_return()

            if not proc and not func:
                raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

            self._returned = True
            self.retval = intp.retval

    def visit_while_stmt(self, stmt: WhileStatement):
        cond: Expr = stmt.cond # type: ignore

        block: list[Statement] = stmt.while_s.block # type: ignore

        intp = self.new(block, loop=True)
        intp.variables = self.variables # scope
        intp.functions = self.functions

        while self.visit_expr(cond).boolean:
            intp.visit_block(block)
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

                self._returned = True
                self.retval = intp.retval
                return

    def visit_for_stmt(self, stmt: ForStatement):
        begin = self.visit_expr(stmt.begin)

        if begin.kind != "integer":
            raise BCRuntimeError("non-integer expression used for for loop begin")

        end = self.visit_expr(stmt.end)

        if end.kind != "integer":
            raise BCRuntimeError("non-integer expression used for for loop end")
        
        if stmt.step is None:
            step = 1
        else:
            step = self.visit_expr(stmt.step).get_integer()
       
        intp = self.new(stmt.block, loop=True)
        intp.calls = self.calls
        intp.calls.append("for")
        intp.variables = self.variables
        intp.functions = self.functions

        counter = begin
        intp.variables[stmt.counter.ident] = Variable(counter, const=False)

        if step > 0:
            while counter.get_integer() <= end.get_integer():
                intp.visit_block(None)
                if intp._returned:
                    proc, func = self.can_return()

                    if not proc and not func:
                        raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

                    self._returned = True
                    self.retval = intp.retval
                    return

                counter.integer = counter.integer + step # type: ignore
        elif step < 0:
            while counter.get_integer() >= end.get_integer():
                intp.visit_block(None)

                if intp._returned:
                    proc, func = self.can_return()

                    if not proc and not func:
                        raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

                    self._returned = True
                    self.retval = intp.retval
                    return

                counter.integer = counter.integer + step # type: ignore

    def visit_repeatuntil_stmt(self, stmt: RepeatUntilStatement):
        cond: Expr = stmt.cond # type: ignore
        intp = self.new(stmt.block, loop=True)
        intp.calls = self.calls
        intp.calls.append("repeatuntil")
        intp.variables = self.variables
        intp.functions = self.functions

        while True:
            intp.visit_block(None)
            if intp._returned:
                proc, func = self.can_return()

                if not proc and not func:
                    raise BCRuntimeError(f"did not find function or procedure to return from! you cannot return from a {self.calls[len(self.calls)-1]}")

                self._returned = True
                self.retval = intp.retval
                return

            if self.visit_expr(cond).boolean:
                break

    def visit_procedure(self, stmt: ProcedureStatement):
        self.functions[stmt.name] = stmt

    def visit_function(self, stmt: FunctionStatement):
        self.functions[stmt.name] = stmt

    def visit_stmt(self, stmt: Statement):
        match stmt.kind:
            case "if":
                self.visit_if_stmt(stmt.if_s) # type: ignore
            case "for":
                self.visit_for_stmt(stmt.for_s) # type: ignore
            case "while":
                self.visit_while_stmt(stmt.while_s) # type: ignore
            case "repeatuntil":
                self.visit_repeatuntil_stmt(stmt.repeatuntil) # type: ignore
            case "output":
                self.visit_output_stmt(stmt.output) # type: ignore
            case "input":
                self.visit_input_stmt(stmt.input) # type: ignore
            case "return":
                self.visit_return_stmt(stmt.return_s) # type: ignore
            case "procedure":
                self.visit_procedure(stmt.procedure) # type: ignore
            case "function":
                self.visit_function(stmt.function) # type: ignore
            case "call":
                self.visit_call(stmt.call) # type: ignore
            case "fncall":
                self.visit_fncall(stmt.fncall) # type: ignore
            case "assign":
                s: AssignStatement = stmt.assign # type: ignore

                if isinstance(s.ident, ArrayIndex):
                    key = s.ident.ident.ident

                    if self.variables[key].val.array is None:
                        raise BCRuntimeError(f"tried to index a variable of type {self.variables[key].val.kind} like an array")

                    tup = self._get_array_index(s.ident)
                    if tup[1] is None and self.variables[key].val.array.typ.is_matrix: # type: ignore
                        raise BCRuntimeError(f"not enough indices for matrix")
                    
                    val = self.visit_expr(s.value)
                    a: BCArray = self.variables[key].val.array # type: ignore
                    
    
                    if a.typ.is_matrix: # type: ignore
                        if tup[0] not in range(a.matrix_bounds[0], a.matrix_bounds[1]+1): # type: ignore
                            raise BCRuntimeError(f"tried to access out of bounds array index {tup[0]}")   
                        
                        if tup[1] not in range(a.matrix_bounds[2], a.matrix_bounds[3]+1): # type: ignore
                            raise BCRuntimeError(f"tried to access out of bounds array index {tup[1]}")

                        a.matrix[tup[0]-1][tup[1]-1] = val # type: ignore
                    else:
                        if tup[0] not in range(a.flat_bounds[0], a.flat_bounds[1]+1): # type: ignore
                            raise BCRuntimeError(f"tried to access out of bounds array index {tup[0]}")
                        a.flat[tup[0]-1] = val # type: ignore
                else:
                    key = s.ident.ident

                    if key not in self.variables:
                        raise BCRuntimeError(f"attempted to use variable {key} without declaring it")

                    if self.variables[key].const:
                        raise BCRuntimeError(f"attemped to write to constant {key}")
                    self.variables[key].val = self.visit_expr(s.value)
            case "constant":
                c: ConstantStatement = stmt.constant # type: ignore
                key = c.ident.ident

                if key in self.variables:
                    raise BCRuntimeError(f"variable {key} declared!")

                self.variables[key] = Variable(c.value.to_bcvalue(), True)
            case "declare":
                d: DeclareStatement = stmt.declare # type: ignore
                key = d.ident.ident
                
                if key in self.variables:
                    raise BCRuntimeError(f"variable {key} declared!") 

                if isinstance(d.typ, BCArrayType):
                    atype = d.typ
                    inner_type = atype.inner
                    if atype.is_matrix:
                        inner_end = self.visit_expr(atype.matrix_bounds[3]) # type: ignore
                        if inner_end.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {inner_end.kind} as array bound!")
                        
                        # directly setting the result of the comprehension results in multiple pionters pointing to the same list
                        get_inner_arr = lambda: [BCValue.empty(inner_type) for _ in range(inner_end.integer)] # type: ignore

                        outer_end = self.visit_expr(atype.matrix_bounds[1]) # type: ignore
                        if outer_end.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {outer_end.kind} as array bound!")

                        outer_arr: BCValue = [get_inner_arr() for _ in range(outer_end.integer)] # type: ignore
                        
                        outer_begin = self.visit_expr(atype.matrix_bounds[0]) # type: ignore
                        if outer_begin.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {outer_begin.kind} as array bound!")
 
                        inner_begin = self.visit_expr(atype.matrix_bounds[2]) # type: ignore
                        if inner_begin.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {inner_begin.kind} as array bound!")

                        bounds = (outer_begin.integer, outer_end.integer, inner_begin.integer, inner_end.integer) # type: ignore
                        atype.is_matrix = True
                        res = BCArray(typ=atype, matrix=outer_arr, matrix_bounds=bounds) # type: ignore
                    else:
                        begin = self.visit_expr(atype.flat_bounds[0]) # type: ignore
                        if begin.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {begin.kind} as array bound!")

                        end = self.visit_expr(atype.flat_bounds[1]) # type: ignore
                        if end.kind != "integer":
                            raise BCRuntimeError(f"cannot use type of {end.kind} as array bound!")

                        arr: BCValue = [BCValue.empty(atype) for _ in range(end.integer)] # type: ignore

                        bounds = (begin.integer, end.integer) # type: ignore
                        atype.is_matrix = False
                        res = BCArray(typ=atype, flat=arr, flat_bounds=bounds) # type: ignore


                    self.variables[key] = Variable(BCValue(kind=d.typ, array=res), False)
                else:
                    self.variables[key] = Variable(BCValue(kind=d.typ), False)

    def visit_block(self, block: list[Statement] | None):
        blk = block if block is not None else self.block
        for stmt in blk:
            self.visit_stmt(stmt)

            if self._returned:
                return

    def visit_program(self, program: Program):
        if program is not None:
            self.visit_block(program.stmts)
