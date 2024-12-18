const std = @import("std");
const lexer = @import("./lexer.zig");
const util = @import("./util.zig");

const ast = @import("./ast/ast.zig");

const Token = lexer.Token;

const Parser = struct {
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

    pub fn init(alloc: std.mem.Allocator, tokens: std.Arraylist(Token)) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
        };
    }

    pub fn arithmeticExpr(self: *Parser) ast.expr.BinaryExpr {
        _ = self;
    }

    pub fn expr(self: *Parser) Expr {
        return self.arithmeticExpr();
    }

    pub fn parse(self: *Parser) Expr {
        return self.expr();
    }
};
