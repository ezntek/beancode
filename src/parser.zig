const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast_types.zig");
const util = @import("util.zig");

const Token = lexer.Token;
const Location = util.Location;
const diag = util.diag;

pub const Parser = struct {
    alloc: std.mem.Allocator,
    tokens: []Token,
    cur: u32,
    loc: Location,

    const Self = @This();

    pub fn prev(self: *const Self) Token {
        return self.tokens[self.cur - 1];
    }

    pub fn peek(self: *const Self) Token {
        return self.tokens[self.cur];
    }

    pub fn peek_next(self: *const Self) Token {
        return self.tokens[self.cur + 1];
    }

    pub fn consume(self: *Self) Token {
        if (self.cur < self.tokens.len) {
            self.cur += 1;
        }

        return self.prev();
    }

    pub fn consume_newlines(self: *Self) void {
        while (self.peek() == .newline) {
            self.consume();
        }
    }

    pub fn check_newline(self: *const Self, ctx: []const u8) void {
        if (self.consume() != .newline) {}
    }

    pub fn init(alloc: std.mem.Allocator, tokens: []lexer.Token) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
            .cur = 0,
        };
    }

    pub fn unary(self: *const Self) ast.Expr {
        const p = self.peek();
    }

    pub fn mathPow(self: *const Self) ast.Expr {}

    pub fn mathMulDiv(self: *const Self) ast.Expr {}

    pub fn mathAddSub(self: *const Self) ast.Expr {}

    pub fn comparison(self: *const Self) ast.Expr {}

    pub fn equality(self: *const Self) ast.Expr {
        const left = self.comparison();
    }

    pub fn logicalComparison(self: *const Self) ast.Expr {
        const left = self.equality();
    }

    pub fn expression(self: *const Self) ast.Expr {
        return self.logicalComparison();
    }

    pub fn program(self: *const Self) ast.Program {
        _ = self;
        return ast.Program{
            // slay
            .stmts = &.{},
        };
    }
};
