const std = @import("std");
const lexer = @import("./lexer.zig");
const util = @import("./util.zig");

const ast = @import("./ast/ast.zig");

const Token = lexer.Token;

pub const Parser = struct {
    alloc: std.mem.Allocator,
    tokens: std.ArrayList(Token),
    precedences: std.AutoHashMap(Operator, u8),
    cur: u32,

    const Expr = ast.expr.Expr;
    const Operator = ast.expr.Operator;

    fn makePrecedenceTable(self: *Parser) void {
        var p = std.AutoHashMap(Operator, u8).init(self.alloc);

        // arithmetic operators
        p.put(Operator{ .arithmetic = .op_add }, 3);
        p.put(Operator{ .arithmetic = .op_sub }, 3);
        p.put(Operator{ .arithmetic = .op_mul }, 4);
        p.put(Operator{ .arithmetic = .op_div }, 4);

        // comparison operators
        p.put(Operator{ .comparison = .op_gt }, 2);
        p.put(Operator{ .comparison = .op_lt }, 2);
        p.put(Operator{ .comparison = .op_geq }, 2);
        p.put(Operator{ .comparison = .op_leq }, 2);
        p.put(Operator{ .comparison = .op_neq }, 2);
        p.put(Operator{ .comparison = .op_eq }, 2);

        // logical operators
        p.put(Operator{ .logical = .op_not }, 1);
        p.put(Operator{ .logical = .op_and }, 1);
        p.put(Operator{ .logical = .op_or }, 1);

        self.precedences = p;
    }

    fn getPrecedence(self: *const Parser, operator: Operator) u8 {
        const res = self.precedences.get(operator);
        return res.?;
    }

    fn advance(self: *Parser) Token {
        const res = self.tokens.items[self.cur];
        self.cur += 1;
        return res;
    }

    fn next(self: *const Parser) Token {
        return self.tokens.items[self.cur + 1];
    }

    fn previous(self: *const Parser) Token {
        return self.tokens.items[self.cur - 1];
    }

    fn literal(self: *Parser) ast.literal.Literal {
        const tok = self.advance();
        const Literal = ast.literal.Literal;

        switch (tok) {
            .literal => |lit| {
                return switch (lit) {
                    .string => |s| Literal{ .string = s },
                    .char => |ch| Literal{ .char = ch },
                    .boolean => |b| Literal{ .boolean = b },
                    .number => |num| {
                        if (std.fmt.parseInt(i32, num, 10)) |payload| {
                            return Literal{ .integer = payload };
                        } else |_| {
                            if (std.fmt.parseFloat(f64, num)) |payload| {
                                return Literal{ .real = payload };
                            } else |_| {
                                util.fatal("parser", "numeric literal `{s}` is neither a REAL nor INTEGER!", .{num});
                            }
                        }
                    },
                };
            },
            else => {
                util.fatal("parser", "expected literal but found {any}", .{tok});
            },
        }
    }

    fn arithmeticOp(self: *Parser) ast.expr.ArithmeticOperator {
        const ArithmeticOperator = ast.expr.ArithmeticOperator;
        const tok = self.advance();

        switch (tok) {
            .operator => |op| switch (op) {
                .add => return ArithmeticOperator.op_add,
                .sub => return ArithmeticOperator.op_sub,
                .mul => return ArithmeticOperator.op_mul,
                .div => return ArithmeticOperator.op_div,
                else => {
                    util.fatal("parser", "expected arithmetic operator but found {any}", .{tok});
                },
            },
            else => util.fatal("parser", "expected an operator but found {any}", .{tok}),
        }
    }

    fn arithmeticExpr(self: *Parser) ast.expr.BinaryExpr {
        const res = ast.expr.BinaryExpr{
            .lhs = self.literal(),
            .op = ast.expr.Operator{ .arithmetic = self.arithmeticOp() },
            .rhs = self.literal(),
        };
        return res;
    }

    fn expr(self: *Parser) Expr {
        return Expr{ .binary = self.arithmeticExpr() };
    }

    pub fn init(alloc: std.mem.Allocator, tokens: std.ArrayList(Token)) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
            .precedences = undefined,
            .cur = 0,
        };
    }

    pub fn parse(self: *Parser) Expr {
        return self.expr();
    }
};
