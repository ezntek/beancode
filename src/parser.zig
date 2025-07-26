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
const panic = util.panic;

const Expr = ast.Expr;

const MAX_ERROR_COUNT = 20;

pub const Parser = struct {
    alloc: std.mem.Allocator,
    file_name: []const u8,
    tokens: []Token,
    cur: u32,
    error_count: u32,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, tokens: []lexer.Token) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
            .cur = 0,
        };
    }

    fn prev(self: *const Self) ?Token {
        return if (self.cur - 1 < self.tokens.len) self.tokens[self.cur - 1] else null;
    }

    fn peek(self: *const Self) ?Token {
        return if (self.cur < self.tokens.len) self.tokens[self.cur] else null;
    }

    fn peek_next(self: *const Self) ?Token {
        return if (self.cur + 1 < self.tokens.len) self.tokens[self.cur + 1] else null;
    }

    fn getLoc(self: *const Self) Location {
        return self.peek().?.loc; // FIXME: better null handling
    }

    fn diag(self: *Self, comptime fmt: []const u8, fmtargs: anytype) void {
        const fname = self.file_name orelse "(no file)";
        util.diag(self.getLoc(), fname, fmt, fmtargs);
        self.bumpErrorCount();
    }

    fn bumpErrorCount(self: *Self) bool {
        self.error_count += 1;
        return self.error_count > MAX_ERROR_COUNT;
    }

    fn consume(self: *Self) ?Token {
        if (self.cur < self.tokens.len) {
            self.cur += 1;
        }

        return self.prev();
    }

    fn consumeAndExpect(self: *Self, expected: TokenKind) ?Token {
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

    fn consumeAndCheck(self: *Self, expected: TokenKind) ?Token {
        if (self.consume()) |tok| {
            if (tok.data == expected) {
                return tok;
            }
        }
        return null;
    }

    fn consumeNewlines(self: *Self) void {
        while (self.peek() == .newline) {
            self.consume();
        }
    }

    fn checkNewline(self: *const Self, ctx: []const u8) void {
        const tok = self.consume();
        if (tok != .newline) {
            const ts = tok.getTokenString(self.alloc);
            defer self.alloc.free(ts);
            self.diag("expected newline after {s} but found `{s}`", .{ ctx, tok });
        }
    }

    fn isInt(val: []const u8) bool {
        var actual = val;
        if (val[0] == '-' and std.ascii.isDigit(val[1])) {
            actual = actual[1..];
        }

        for (actual) |ch| {
            if (!std.ascii.isDigit(ch) and ch != '_') {
                return false;
            }
        }

        return true;
    }

    fn isFloat(val: []const u8) bool {
        if (isInt(val)) return false;

        var found = false;
        for (val) |ch| {
            if (ch == '.') {
                if (found) break false;
                found = true;
            }
        }
        return found;
    }

    fn primitive(self: *Self) ?ast.Expr {
        const tok = self.consumeAndCheck(.primitive) orelse return null;
        switch (tok.data.primitive) {
            .char => |val| {
                if (val[0] == '\\') {
                    const res: u8 = switch (val[1]) {
                        'n' => 0x0a,
                        'r' => 0x0d,
                        't' => 0x09,
                        'a' => 0x07,
                        'b' => 0x08,
                        'f' => 0x0c,
                        'v' => 0x0b,
                        'e' => 0x1b,
                        '\\' => 0x5c,
                        else => {
                            self.diag("invalid escape sequence '\\{c}'", .{val[1]});
                            return null;
                        },
                    };
                    return Expr.makePrimitive(self.alloc, .{ .char = res }, self.getLoc());
                } else if (val.len > 1) {
                    self.diag("more than one character in char literal `{s}`", .{val});
                    return null;
                } else {
                    return Expr.makePrimitive(self.alloc, .{ .char = val[0] }, self.getLoc());
                }
            },
            .string => |val| {
                const s = self.alloc.dupe(u8, val) catch |err| panic(err);
                return Expr.makePrimitive(self.alloc, .{ .string = s }, self.getLoc());
            },
            .bool => |val| {
                var prim = undefined;
                if (std.ascii.eqlIgnoreCase(val, "true")) {
                    prim = .{ .bool = true };
                } else if (std.ascii.eqlIgnoreCase(val, "false")) {
                    prim = .{ .bool = false };
                } else {
                    self.diag("invalid boolean literal \"{s}\"", .{val});
                    return null;
                }
                return Expr.makePrimitive(self.alloc, prim, self.getLoc());
            },
            .number => |num| {
                if (isFloat(num)) {
                    const res = std.fmt.parseFloat(f64, num) catch |err| {
                        self.diag("invalid float literal \"{s}\": \"{any}\"", .{ num, err });
                        return null;
                    };
                    return Expr.makePrimitive(self.alloc, .{ .float = res }, self.getLoc());
                } else if (isInt(num)) {
                    const res = std.fmt.parseInt(i32, num, 10) catch |err| { // TODO: support other bases
                        self.diag("invalid int literal \"{s}\": \"{any}\"", .{ num, err });
                        return null;
                    };
                    return Expr.makePrimitive(self.alloc, .{ .int = res }, self.getLoc());
                } else {
                    self.diag("found invalid number literal \"{s}\"", .{num});
                }
            },
        }
    }

    fn ident(self: *const Self) Expr {
        // ident checking logic was already done
        // we know for sure this is an ident
        const p = self.consume().?;

        return Expr.makeIdent(self.alloc, p.data.ident, self.getLoc());
    }

    fn functionCall(self: *const Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn lvalueArrayIndex(self: *const Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn typecast(self: *const Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn unary(self: *const Self) ?ast.Expr {
        const p = self.peek();
        switch (p.data) {
            .primitive => |_| return self.primitive(),
            .ident => |_| {
                const next = self.peek_next() orelse return self.ident();
                if (next.data == .separator and next.data.separator == .left_paren) {
                    return self.functionCall();
                } else if (next.data == .separator and next.data.separator == .left_bracket) {
                    return self.lvalueArrayIndex();
                } else {
                    return self.ident();
                }
            },
            .type => |_| {
                const next = self.peek_next() orelse return null;
                if (next.data == .separator and next.data.separator == .left_paren) {
                    return self.typecast();
                }
            },
            // TODO: implement others
            else => return null,
        }
        return null;
    }

    fn mathPow(self: *const Self) ?ast.Expr {
        return self.unary();
    }

    fn mathMulDiv(self: *const Self) ?ast.Expr {
        return self.mathPow();
    }

    fn mathAddSub(self: *const Self) ?ast.Expr {
        return self.mathMulDiv();
    }

    fn comparison(self: *const Self) ?ast.Expr {
        return self.mathAddSub();
    }

    fn equality(self: *const Self) ?ast.Expr {
        const left = self.comparison();
        return left;
    }

    fn logicalComparison(self: *const Self) ?ast.Expr {
        const left = self.equality();
        return left;
    }

    pub fn expression(self: *const Self) ?ast.Expr {
        return self.logicalComparison();
    }

    fn printStmt(self: *const Self) ?ast.Statement {
        const begin = self.peek() orelse return null;
        if (begin.data != .keyword) return null;
        if (begin.data.keyword != .kw_print) return null;

        _ = self.consume().?;

        const initial = self.expression();
        if (initial) |exp| {
            _ = exp; // FIXME:
        }
    }

    pub fn program(self: *const Self) ast.Program {
        _ = self;
        return ast.Program{
            // slay
            .stmts = &.{},
        };
    }
};
