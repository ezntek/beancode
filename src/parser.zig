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
const Type = ast.Type;
const Lvalue = ast.Lvalue;
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

    fn get(self: *const Self, pt: u32) ?Token {
        return if (self.cur < self.tokens.len) self.tokens[pt] else null;
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

    fn peekNextAndCheck(self: *const Self, comptime wanted: TokenKind) bool {
        if (self.peekNext()) |tok| {
            return tok.data == wanted;
        } else {
            return false;
        }
    }

    fn peekNextAndExpect(self: *Self, comptime expected: TokenKind) ?Token {
        if (self.peekNext()) |tok| {
            if (tok.data != expected) {
                self.diag("expected token {s} but got {s}", .{ expected, tok });
            } else {
                return tok;
            }
        } else {
            self.diag("expected token {s} but reached the end of the token stream", .{expected});
        }
        return null;
    }

    fn getSpan(self: *const Self) *const SourceSpan {
        // XXX: I dont know if this is supposed to work like this!
        return &self.prev().?.span; // FIXME: better null handling
    }

    fn check(self: *const Self, comptime tok: TokenKind) bool {
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

    fn skip(self: *Self) void {
        _ = self.consume();
    }

    fn consumeAndExpect(self: *Self, expected: TokenKind) ?Token {
        if (self.consume()) |tok| {
            if (tok.data != expected) {
                self.diag("expected token \"{s}\" but got \"{s}\"", .{ expected, tok });
            } else {
                return tok;
            }
        } else {
            self.diag("expected token \"{s}\" but reached the end of the token stream", .{expected});
        }
        return null;
    }

    fn match(self: *Self, comptime vals: anytype) ?Token {
        inline for (vals) |val| {
            if (self.check(val)) {
                // we dont care about this value
                return self.consume();
            }
        }
        return null;
    }

    fn diagAtSpan(self: *Self, span: *const SourceSpan, comptime fmt: []const u8, fmtargs: anytype) void {
        const fname = self.file_name orelse "(no file)";
        util.diag(span, fname, fmt, fmtargs);
        _ = self.bumpErrorCount();
        // TODO: proper error handling
    }

    fn diag(self: *Self, comptime fmt: []const u8, fmtargs: anytype) void {
        const a = self.getSpan().*; // this works, dont touch
        return self.diagAtSpan(&a, fmt, fmtargs);
    }

    fn diagExpected(self: *Self, comptime msg: []const u8) void {
        if (self.peek()) |pk| {
            self.diag("expected {s}, but found \"{s}\"", .{ msg, pk });
        } else {
            // XXX: is this really a good message?
            self.diag("expected {s}, but found no token", .{msg});
        }
    }

    fn diagAndSkip(self: *Self, comptime msg: []const u8, fmtargs: anytype) void {
        self.diag(msg, fmtargs);
        self.skip();
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
        while (self.peek()) |tok| {
            if (tok.data == .newline) {
                _ = self.consume();
            } else {
                break;
            }
        }
    }

    fn checkNewline(self: *Self, ctx: []const u8) void {
        const tok = self.consume() orelse return; // end of stream should be fine
        if (tok.data != .newline and tok.data != .eof) {
            self.diag("expected newline after {s} but found `{s}`", .{ ctx, tok });
        }
    }

    fn primitiveType(self: *Self) ?Type {
        const tok = self.consume() orelse return null;
        return switch (tok.data) {
            .t_int => Type.initPrimitive(.int),
            .t_float => Type.initPrimitive(.float),
            .t_bool => Type.initPrimitive(.bool),
            .t_string => Type.initPrimitive(.string),
            .t_char => Type.initPrimitive(.char),
            else => null,
        };
    }

    // i love shadowing names to not piss the compiler off
    fn _type(self: *Self) ?ast.Type {
        return self.primitiveType(); // TODO: array type
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

    /// Get an identifier as a slice
    fn identRaw(self: *Self) ?[]const u8 {
        const p = self.peekAndCheckThenConsume(.ident) orelse return null;
        return p.data.ident;
    }

    /// Get an identifier as an expression
    fn ident(self: *Self) ?Expr {
        const id_raw = self.identRaw() orelse return null;
        return Expr.initIdent(self.alloc, id_raw, self.getSpan());
    }

    fn lvalueArrayIndex(self: *Self) ?Expr {
        _ = self;
        unreachable;
    }

    fn lvalueIdent(self: *Self) ?ast.Lvalue {
        const id = self.consumeAndCheck(.ident) orelse return null;
        return Lvalue.initIdent(self.alloc, id.data.ident);
    }

    fn lvalue(self: *Self) ?ast.Lvalue {
        return self.lvalueIdent();
    }

    fn functionCall(self: *Self) ?Expr {
        // TODO: implement
        _ = self;
        unreachable;
    }

    fn typecast(self: *Self, typ: TokenData) ?Expr {
        // this line shouldn't actually return null, because we check it in
        // unary()
        const prim = ast.PrimitiveType.fromToken(typ) orelse return null;

        const bracket = self.peekNextAndExpect(.left_paren) orelse return null;

        const inner = self.expression() orelse {
            self.diagAndSkip("found invalid expression within typecast", .{});
            return null;
        };

        return Expr.initTypecast(self.alloc, prim, inner, &bracket.span);
    }

    fn unary(self: *Self) ?ast.Expr {
        const p = self.peek() orelse return null;
        switch (p.data) {
            .primitive => |_| return self.primitive(),
            .ident => |_| {
                const next = self.peekNext() orelse {
                    return self.ident() orelse return null;
                };

                if (next.data == .left_paren) {
                    return self.functionCall();
                } else if (next.data == .left_bracket) {
                    return self.lvalueArrayIndex();
                } else {
                    return self.ident();
                }
            },
            .t_int, .t_float, .t_bool, .t_string, .t_char => {
                if (self.peekNextAndCheck(.left_paren)) {
                    return self.typecast(p.data);
                }
            },
            .left_paren => {
                const begin = self.consume() orelse return null;
                const expr = self.expression() orelse {
                    self.diagAndSkip("invalid expression within grouping", .{});
                    return null;
                };

                _ = self.consumeAndExpect(.right_paren) orelse return null;

                // FIXME: span is incorrect
                return Expr.initGrouping(
                    self.alloc,
                    expr,
                    &begin.span,
                );
            },
            // TODO: implement others
            else => return null,
        }
        return null;
    }

    fn mathPow(self: *Self) ?ast.Expr {
        var base = self.unary() orelse return null;

        while (self.match(.{.pow})) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.unary() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    fn mathMulDiv(self: *Self) ?ast.Expr {
        var base = self.mathPow() orelse return null;

        while (self.match(.{ .mul, .div })) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.mathPow() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    fn mathAddSub(self: *Self) ?ast.Expr {
        var base = self.mathMulDiv() orelse return null;

        while (self.match(.{ .sub, .add })) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.mathMulDiv() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    fn comparison(self: *Self) ?ast.Expr {
        var base = self.mathAddSub() orelse return null;

        const itms = .{ .greater_than, .greater_than_or_equal, .less_than, .less_than_or_equal };
        while (self.match(itms)) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.mathAddSub() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    fn equality(self: *Self) ?ast.Expr {
        var base = self.comparison() orelse return null;

        while (self.match(.{ .equal, .not_equal })) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.comparison() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    fn logicalComparison(self: *Self) ?ast.Expr {
        var base = self.equality() orelse return null;

        while (self.match(.{ .k_and, .k_or })) |op_tok| {
            const op = ast.Operator.fromToken(op_tok.data).?;
            const right = self.equality() orelse return null;
            base = ast.Expr.initBinary(self.alloc, base, op, right, &op_tok.span);
        }

        return base;
    }

    pub fn expression(self: *Self) ?ast.Expr {
        return self.logicalComparison();
    }

    fn printStmt(self: *Self) ?ast.Statement {
        const begin = self.peekAndCheckThenConsume(.k_print) orelse return null;

        var exprs: std.ArrayListUnmanaged(Expr) = .empty;
        defer exprs.deinit(self.alloc);

        while (self.expression()) |exp| {
            exprs.append(self.alloc, exp) catch |err| panic(err);

            if (self.consume()) |tok| {
                switch (tok.data) {
                    .comma => {},
                    .newline, .eof => break,
                    else => self.diag("found invalid token after expression in print: \"{s}\"", .{tok}),
                }
            }
        } else {
            self.diag("found invalid expression after print keyword", .{});
            _ = self.consume(); // get past the issue
            return null;
        }

        // we check inside the while loop
        // self.checkNewline("print statement");

        return Statement.initPrintStmt(self.alloc, exprs.items, &begin.span);
    }

    fn readStmt(self: *Self) ?ast.Statement {
        const begin = self.peekAndCheckThenConsume(.k_read) orelse return null;

        const id = self.lvalue() orelse {
            self.diagAndSkip("found invalid expression after read keyword", .{});
            return null;
        };

        self.checkNewline("read statement");

        return Statement.initReadStmt(self.alloc, id, &begin.span);
    }

    fn constStmt(self: *Self) ?ast.Statement {
        const begin = self.peekAndCheckThenConsume(.k_const) orelse return null;

        const id = self.identRaw() orelse {
            self.diagExpected("identifier after const");
            return null;
        };

        _ = self.consumeAndExpect(.assign) orelse return null;

        const exp = self.expression() orelse {
            self.diagAndSkip("found invalid expression after assignment in const statement", .{});
            return null;
        };

        self.checkNewline("const statement");

        // TODO: export support
        return Statement.initConstStmt(self.alloc, id, exp, false, &begin.span);
    }

    fn varStmt(self: *Self) ?ast.Statement {
        const begin = self.peekAndCheckThenConsume(.k_var) orelse return null;
        var val: ?Expr = null;
        var typ: ?Type = null;

        // === ALLOWED ===
        // var i: int = 3
        // var j: int
        // var x = 4;
        // === DISALLOWED ===
        // var x

        const id = self.identRaw() orelse {
            self.diagExpected("ident after var");
            return null;
        };

        if (self.peekAndCheckThenConsume(.colon)) |_| {
            if (self._type()) |t| {
                typ = t;
            } else {
                self.diagAndSkip("found invalid type in var statement", .{});
            }
        }

        if (self.peekAndCheckThenConsume(.assign)) |_| {
            val = self.expression() orelse {
                self.diagAndSkip("found invalid expression in var statement", .{});
                return null;
            };
        }

        if (val == null and typ == null) {
            self.diag("either a value, type or both must be supplied to a var statement!", .{});
            return null;
        }

        self.checkNewline("var statement");

        return Statement.initVarStmt(self.alloc, id, typ, val, false, &begin.span);
    }

    fn assignStmt(self: *Self) ?ast.Statement {
        const saved_point = self.cur;
        const initial = self.lvalue();

        const ass = self.peekAndCheckThenConsume(.assign) orelse {
            self.cur = saved_point;
            return null;
        };

        const lv = initial orelse {
            self.diag("invalid left hand side expression in assignment", .{});
            return null;
        };

        const exp = self.expression() orelse {
            self.diag("found invalid expression in assignment", .{});
            return null;
        };

        self.checkNewline("assignment");

        return Statement.initAssignStmt(lv, exp, &ass.span);
    }

    fn statement(self: *Self) ?Statement {
        self.consumeNewlines();

        if (self.printStmt()) |v| return v;
        if (self.readStmt()) |v| return v;
        if (self.constStmt()) |v| return v;
        if (self.varStmt()) |v| return v;
        if (self.assignStmt()) |v| return v;

        return null;
    }

    fn skipToNextNewline(self: *Self) void {
        while (self.peek()) |pk| {
            if (pk.data == .newline or pk.data == .eof) {
                return;
            } else {
                self.skip();
            }
        }
    }

    fn nextStatement(self: *Self) ?Statement {
        const s = self.statement();
        if (s) |res| {
            self.consumeNewlines();
            return res;
        } else {
            self.skipToNextNewline();
            return null;
        }
    }

    pub fn program(self: *Self) ast.Program {
        var stmts: std.ArrayListUnmanaged(Statement) = .empty;
        defer stmts.deinit(self.alloc);

        while (self.cur < self.tokens.len) {
            if (self.peek()) |pk| {
                if (pk.data == .eof) {
                    break;
                }
            }

            self.consumeNewlines();
            if (self.nextStatement()) |stmt| {
                stmts.append(self.alloc, stmt) catch |err| panic(err);
            } else {
                // we might put other things here
                if (self.error_count > MAX_ERROR_COUNT) {
                    util.fatal("parser", "too many errors emitted!", .{});
                }
            }
        }

        if (self.error_count > 1) {
            util.fatal("parser", "emitted {} errors.", .{self.error_count});
        } else if (self.error_count == 1) {
            util.fatal("parser", "emitted 1 error.", .{});
        }

        const res = ast.Program.init(self.alloc, stmts.items);
        return res;
    }
};
