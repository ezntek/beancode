const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast_types.zig");
const util = @import("util.zig");
const common = @import("common.zig");

const Token = lexer.Token;
const TokenData = lexer.TokenData;
const TokenKind = lexer.TokenKind;
const SourceSpan = util.SourceSpan;
const diag = util.diag;
const panic = util.panic;

const Expr = ast.Expr;
const Statement = ast.Statement;

const MAX_ERROR_COUNT = 20;

pub const Parser = struct {
    alloc: std.mem.Allocator,
    file_name: ?[]const u8,
    tokens: []Token,
    cur: u32,
    error_count: u32,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, file_name: ?[]const u8, tokens: []lexer.Token) Parser {
        var res = Parser{
            .alloc = alloc,
            .file_name = null,
            .tokens = tokens,
            .cur = 0,
            .error_count = 0,
        };
        if (file_name) |name| {
            const fname = alloc.dupe(u8, name) catch |err| panic(err);
            res.file_name = fname;
        }
        return res;
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        if (self.file_name) |n| {
            alloc.free(n);
        }
    }

    fn prev(self: *const Self) ?Token {
        return if (self.cur - 1 < self.tokens.len) self.tokens[self.cur - 1] else null;
    }

    fn peek(self: *const Self) ?Token {
        return if (self.cur < self.tokens.len) self.tokens[self.cur] else null;
    }

    fn peekAndExpect(self: *const Self, comptime expected: TokenKind) ?Token {
        if (self.peek()) |tok| {
            if (tok.data != expected) {
                self.diag("expected token {s} but got {s}", .{ expected, tok });
            } else {
                return tok;
            }
        } else {
            self.diag("expected token {s} but reached the end of the token stream", .{expected});
        }
    }

    fn peekAndCheck(self: *const Self, comptime wanted: TokenKind) bool {
        if (self.peek()) |tok| {
            return tok.data == wanted;
        } else {
            return false;
        }
    }

    fn peekAndCheckThenConsume(self: *Self, comptime wanted: TokenKind) ?Token {
        const res = self.peekAndCheck(wanted);
        if (res) {
            return self.consume();
        } else {
            return null;
        }
    }

    fn peekNext(self: *const Self) ?Token {
        return if (self.cur + 1 < self.tokens.len) self.tokens[self.cur + 1] else null;
    }

    fn peekNextAndExpect(self: *const Self, comptime expected: TokenKind) ?Token {
        if (self.peekNext()) |tok| {
            if (tok.data != expected) {
                self.diag("expected token {s} but got {s}", .{ expected, tok });
            } else {
                return tok;
            }
        } else {
            self.diag("expected token {s} but reached the end of the token stream", .{expected});
        }
    }

    fn getSpan(self: *const Self) *const SourceSpan {
        return &self.peek().?.span; // FIXME: better null handling
    }

    fn check(self: *const Self, comptime tok: TokenData) bool {
        const p = self.peek() orelse return false;
        return p.data == tok;
    }

    fn checkTokenKind(self: *const Self, comptime kind: TokenKind) bool {
        const p = self.peek() orelse return false;
        return p.data == kind;
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

    fn match(self: *const Self, comptime vals: []TokenData) bool {
        for (vals) |val| {
            if (self.check(val)) {
                // we dont care about this value
                _ = self.consume();
                return true;
            }
        }
        return false;
    }

    fn diag(self: *Self, comptime fmt: []const u8, fmtargs: anytype) void {
        const fname = self.file_name orelse "(no file)";
        util.diag(self.getSpan(), fname, fmt, fmtargs);
        _ = self.bumpErrorCount();
        // TODO: proper error handlinG
    }

    fn bumpErrorCount(self: *Self) bool {
        self.error_count += 1;
        return self.error_count > MAX_ERROR_COUNT;
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
                if (found)
                    return false;
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
                    return Expr.initPrimitive(self.alloc, .char, res, self.getSpan());
                } else if (val.len > 1) {
                    self.diag("more than one character in char literal `{s}`", .{val});
                    return null;
                } else {
                    return Expr.initPrimitive(self.alloc, .char, val[0], self.getSpan());
                }
            },
            .string => |val| {
                const prim = ast.Literal.initPrimitiveString(self.alloc, val);
                return Expr.init(self.alloc, .e_literal, prim, self.getSpan());
            },
            .bool => |val| {
                var res: bool = undefined;
                if (std.ascii.eqlIgnoreCase(val, "true")) {
                    res = true;
                } else if (std.ascii.eqlIgnoreCase(val, "false")) {
                    res = false;
                } else {
                    self.diag("invalid boolean literal \"{s}\"", .{val});
                    return null;
                }
                return Expr.initPrimitive(self.alloc, .bool, res, self.getSpan());
            },
            .number => |num| {
                if (isFloat(num)) {
                    const res = std.fmt.parseFloat(f64, num) catch |err| {
                        self.diag("invalid float literal \"{s}\": \"{any}\"", .{ num, err });
                        return null;
                    };
                    return Expr.initPrimitive(self.alloc, .float, res, self.getSpan());
                } else if (isInt(num)) {
                    const res = std.fmt.parseInt(i32, num, 10) catch |err| { // TODO: support other bases
                        self.diag("invalid int literal \"{s}\": \"{any}\"", .{ num, err });
                        return null;
                    };
                    return Expr.initPrimitive(self.alloc, .int, res, self.getSpan());
                } else {
                    self.diag("found invalid number literal \"{s}\"", .{num});
                }
            },
        }
        return null;
    }

    fn ident(self: *Self) Expr {
        // ident checking logic was already done
        // we know for sure this is an ident
        const p = self.consume().?;

        return Expr.initIdent(self.alloc, p.data.ident, self.getSpan());
    }

    fn functionCall(self: *Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn lvalueArrayIndex(self: *Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn typecast(self: *Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn unary(self: *Self) ?ast.Expr {
        const p = self.peek() orelse return null;
        switch (p.data) {
            .primitive => |_| return self.primitive(),
            .ident => |_| {
                const next = self.peekNext() orelse return self.ident();
                if (next.data == .left_paren) {
                    return self.functionCall();
                } else if (next.data == .left_bracket) {
                    return self.lvalueArrayIndex();
                } else {
                    return self.ident();
                }
            },
            .t_int, .t_float, .t_bool, .t_string, .t_char => {
                const next = self.peekNext() orelse return null;
                if (next.data == .left_paren) {
                    return self.typecast();
                }
            },
            // TODO: implement others
            else => return null,
        }
        return null;
    }

    fn mathPow(self: *Self) ?ast.Expr {
        //const base = self.unary() orelse return null;
        return self.unary();
    }

    fn mathMulDiv(self: *Self) ?ast.Expr {
        return self.mathPow();
    }

    fn mathAddSub(self: *Self) ?ast.Expr {
        return self.mathMulDiv();
    }

    fn comparison(self: *Self) ?ast.Expr {
        return self.mathAddSub();
    }

    fn equality(self: *Self) ?ast.Expr {
        const left = self.comparison();
        return left;
    }

    fn logicalComparison(self: *Self) ?ast.Expr {
        const left = self.equality();
        return left;
    }

    pub fn expression(self: *Self) ?ast.Expr {
        return self.logicalComparison();
    }

    fn printStmt(self: *Self) ?ast.Statement {
        _ = self.peekAndCheckThenConsume(.k_print) orelse return null;

        const initial = self.expression();
        if (initial) |exp| {
            return Statement.initPrintStmt(&.{exp}, self.alloc, self.getSpan());
        } else {
            self.diag("found invalid expression after print keyword", .{});
        }
        return null;
    }

    pub fn program(self: *Self) ast.Program {
        const print = self.printStmt().?;
        const res = ast.Program.init(self.alloc, &.{print});
        return res;
    }
};
