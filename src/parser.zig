const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast_types.zig");
const util = @import("util.zig");
const common = @import("common.zig");

const Token = lexer.Token;
const TokenData = lexer.TokenData;
const TokenKind = lexer.TokenKind;
const Location = util.Location;
const diag = util.diag;

const Expr = ast.Expr;

const MAX_ERROR_COUNT = 20;

pub const Parser = struct {
    alloc: std.mem.Allocator,
    tokens: []Token,
    cur: u32,
    error_count: u32,

    const Self = @This();

    pub fn prev(self: *const Self) ?Token {
        return if (self.cur - 1 < self.tokens.len) self.tokens[self.cur - 1] else null;
    }

    pub fn peek(self: *const Self) ?Token {
        return if (self.cur < self.tokens.len) self.tokens[self.cur] else null;
    }

    pub fn peek_next(self: *const Self) ?Token {
        return if (self.cur + 1 < self.tokens.len) self.tokens[self.cur + 1] else null;
    }

    pub fn getLoc(self: *const Self) Location {
        return self.peek().?.loc; // FIXME: better null handling
    }

    pub fn diag(self: *const Self, comptime fmt: []const u8, fmtargs: anytype) void {
        util.diag(self.getLoc(), fmt, fmtargs);
    }

    pub fn bumpErrorCount(self: *Self) bool {
        self.error_count += 1;
        return self.error_count > MAX_ERROR_COUNT;
    }

    pub fn consume(self: *Self) ?Token {
        if (self.cur < self.tokens.len) {
            self.cur += 1;
        }

        return self.prev();
    }

    pub fn consumeAndExpect(self: *Self, expected: TokenKind) ?Token {
        if (self.consume()) |tok| {
            if (tok.data != expected) {
                self.diag("expected token {s} but got {s}", .{ expected, tok });
            } else {
                return tok;
            }
        } else {
            self.diag("expected token {s} but reached the end of the token stream", .{expected});
        }
    }

    pub fn consumeAndCheck(self: *Self, expected: TokenKind) ?Token {
        if (self.consume()) |tok| {
            if (tok.data == expected) {
                return tok;
            }
        }
        return null;
    }

    pub fn consumeNewlines(self: *Self) void {
        while (self.peek() == .newline) {
            self.consume();
        }
    }

    pub fn checkNewline(self: *const Self, ctx: []const u8) void {
        const tok = self.consume();
        if (tok != .newline) {
            const ts = tok.getTokenString(self.alloc);
            defer self.alloc.free(ts);
            self.diag("expected newline after {s} but found `{s}`", .{ ctx, tok });
        }
    }

    pub fn init(alloc: std.mem.Allocator, tokens: []lexer.Token) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
            .cur = 0,
        };
    }

    pub fn primitive(self: *const Self) ?ast.Expr {
        const tok = self.consumeAndCheck(.primitive) orelse return null;
        switch (tok.data.primitive) {
            .char => |val| {
                if (val[0] == '\\') {
                    Expr.makePrimitive(self.alloc, .{ .char = val[0] });
                }
            },
        }
    }

    pub fn unary(self: *const Self) ?ast.Expr {
        const p = self.peek();
        return switch (p.data) {
            .primitive => |_| self.primitive(),
            else => return null,
        };
    }

    pub fn mathPow(self: *const Self) ?ast.Expr {
        return self.unary();
    }

    pub fn mathMulDiv(self: *const Self) ?ast.Expr {
        return self.mathPow();
    }

    pub fn mathAddSub(self: *const Self) ?ast.Expr {
        return self.mathMulDiv();
    }

    pub fn comparison(self: *const Self) ?ast.Expr {
        return self.mathAddSub();
    }

    pub fn equality(self: *const Self) ?ast.Expr {
        const left = self.comparison();
        return left;
    }

    pub fn logicalComparison(self: *const Self) ?ast.Expr {
        const left = self.equality();
        return left;
    }

    pub fn expression(self: *const Self) ?ast.Expr {
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
