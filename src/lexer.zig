const std = @import("std");
const util = @import("./util.zig");
const ast = @import("./ast_types.zig");

const SourceSpan = util.SourceSpan;
const diag = util.diag;
const panic = util.panic;

pub const PrimitiveKind = enum { number, string, char, bool };

pub const Primitive = union(PrimitiveKind) {
    number: []const u8,
    string: []const u8,
    char: []const u8,
    bool: []const u8,

    const Self = @This();

    pub fn init(comptime kind: PrimitiveKind, alloc: std.mem.Allocator, slc: []const u8) Self {
        const res = alloc.dupe(u8, slc) catch |err| panic(err);
        return @unionInit(Primitive, @tagName(kind), res);
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            // FIXME: what the hell
            .number, .string, .char, .bool => |s| alloc.free(s),
        }
    }
};

pub const TokenKind = enum {
    // keywords
    k_var,
    k_const,
    k_print,
    k_read,
    k_and,
    k_or,
    k_not,
    k_if,
    k_then,
    k_else,
    k_end,
    k_switch,
    k_otherwise,
    k_while,
    k_do,
    k_repeat,
    k_until,
    k_for,
    k_to,
    k_fn,
    k_break,
    k_continue,
    // types
    t_int,
    t_float,
    t_bool,
    t_string,
    t_char,
    // operators
    assign,
    equal,
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,
    not_equal,
    mul,
    div,
    add,
    sub,
    pow,
    mul_assign,
    div_assign,
    add_assign,
    sub_assign,
    pow_assign,
    // separators
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    left_curly,
    right_curly,
    colon,
    comma,
    dot,
    // other
    newline,
    eof,
    ident,
    primitive,
};

pub const TokenData = union(TokenKind) {
    // keywords
    k_var,
    k_const,
    k_print,
    k_read,
    k_and,
    k_or,
    k_not,
    k_if,
    k_then,
    k_else,
    k_end,
    k_switch,
    k_otherwise,
    k_while,
    k_do,
    k_repeat,
    k_until,
    k_for,
    k_to,
    k_fn,
    k_break,
    k_continue,
    // types
    t_int,
    t_float,
    t_bool,
    t_string,
    t_char,
    // operators
    assign,
    equal,
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,
    not_equal,
    mul,
    div,
    add,
    sub,
    pow,
    mul_assign,
    div_assign,
    add_assign,
    sub_assign,
    pow_assign,
    // separators
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    left_curly,
    right_curly,
    colon,
    comma,
    dot,
    // other
    newline,
    eof,
    ident: []const u8,
    primitive: Primitive,

    const Self = @This();

    pub fn init(comptime kind: TokenData, v: anytype) Self {
        return @unionInit(TokenData, @tagName(kind), v);
    }

    pub fn initIdent(alloc: std.mem.Allocator, ident: []const u8) Self {
        const res = alloc.dupe(u8, ident) catch |err| panic(err);
        return Self{
            .ident = res,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .ident => |id| alloc.free(id),
            .primitive => |prim| prim.deinit(alloc),
            else => return,
        }
    }

    pub fn printToken(self: *const Self) void {
        std.debug.print("{s}\n", .{self});
    }

    pub fn getTokenStringAlloc(self: *const Self, alloc: std.mem.Allocator) []const u8 {
        std.fmt.allocPrint(alloc, "{s}", .{self}) catch |err| panic(err);
    }

    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        _ = try switch (self) {
            .ident => |ident| writer.print("ident({s})", .{ident}),
            .primitive => |prim| switch (prim) {
                .string => |s| writer.print("primitive(\"{s}\")", .{s}),
                .number => |n| writer.print("primitive({s})", .{n}),
                .bool => |b| writer.print("primitive({s})", .{b}),
                .char => |ch| writer.print("primitive('{s}')", .{ch}),
            },
            else => {
                const tn = @tagName(self);
                if (tn[1] == '_') {
                    try writer.print("token({s})", .{tn[2..]});
                } else {
                    try writer.print("token({s})", .{tn});
                }
            },
        };
    }
};

pub const Token = struct {
    data: TokenData,
    span: SourceSpan,

    const Self = @This();

    pub fn init(data: TokenData, span: SourceSpan) Self {
        return Self{
            .data = data,
            .span = span,
        };
    }

    pub fn initIdent(alloc: std.mem.Allocator, ident: []const u8, span: SourceSpan) Self {
        const data = TokenData.initIdent(alloc, ident);
        return Self{ .data = data, .span = span };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        return self.data.deinit(alloc);
    }

    pub fn printToken(self: *const Self) void {
        return self.data.printToken();
    }

    pub fn getTokenStringAlloc(self: *const Self, alloc: std.mem.Allocator) []const u8 {
        return self.data.getTokenStringAlloc(alloc);
    }

    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        return self.data.format(fmt, options, writer);
    }
};

pub const Lexer = struct {
    alloc: std.mem.Allocator,
    file_name: []const u8,
    file: []const u8,
    row: u32,
    cur: u32,
    bol: u32,
    keywords: std.StringHashMapUnmanaged(TokenData),
    types: std.StringHashMapUnmanaged(TokenData),
    res: std.ArrayListUnmanaged(Token),

    const Self = @This();

    pub fn init(
        alloc: std.mem.Allocator,
        file: []const u8,
        file_name: []const u8,
    ) Lexer {
        var keywords: std.StringHashMapUnmanaged(TokenData) = .empty;
        keywords.put(alloc, "VAR", .k_var) catch |err| util.panic(err);
        keywords.put(alloc, "CONST", .k_const) catch |err| util.panic(err);
        keywords.put(alloc, "PRINT", .k_print) catch |err| util.panic(err);
        keywords.put(alloc, "READ", .k_read) catch |err| util.panic(err);
        keywords.put(alloc, "AND", .k_and) catch |err| util.panic(err);
        keywords.put(alloc, "OR", .k_or) catch |err| util.panic(err);
        keywords.put(alloc, "NOT", .k_not) catch |err| util.panic(err);
        keywords.put(alloc, "IF", .k_if) catch |err| util.panic(err);
        keywords.put(alloc, "THEN", .k_then) catch |err| util.panic(err);
        keywords.put(alloc, "ELSE", .k_else) catch |err| util.panic(err);
        keywords.put(alloc, "END", .k_end) catch |err| util.panic(err);
        keywords.put(alloc, "SWITCH", .k_switch) catch |err| util.panic(err);
        keywords.put(alloc, "WHILE", .k_while) catch |err| util.panic(err);
        keywords.put(alloc, "DO", .k_do) catch |err| util.panic(err);
        keywords.put(alloc, "REPEAT", .k_repeat) catch |err| util.panic(err);
        keywords.put(alloc, "UNTIL", .k_until) catch |err| util.panic(err);
        keywords.put(alloc, "FOR", .k_for) catch |err| util.panic(err);
        keywords.put(alloc, "TO", .k_to) catch |err| util.panic(err);
        keywords.put(alloc, "CASE", .k_switch) catch |err| util.panic(err);
        keywords.put(alloc, "FN", .k_fn) catch |err| util.panic(err);
        keywords.put(alloc, "BREAK", .k_break) catch |err| util.panic(err);
        keywords.put(alloc, "CONTINUE", .k_continue) catch |err| util.panic(err);

        var types: std.StringHashMapUnmanaged(TokenData) = .empty;
        types.put(alloc, "INT", .t_int) catch |err| util.panic(err);
        types.put(alloc, "FLOAT", .t_float) catch |err| util.panic(err);
        types.put(alloc, "STRING", .t_string) catch |err| util.panic(err);
        types.put(alloc, "CHAR", .t_char) catch |err| util.panic(err);
        types.put(alloc, "BOOLEAN", .t_bool) catch |err| util.panic(err);

        const al: std.ArrayListUnmanaged(Token) = .empty;

        return Lexer{
            .alloc = alloc,
            .file = file,
            .file_name = file_name,
            .cur = 0,
            .bol = 0,
            .row = 1,
            .keywords = keywords,
            .types = types,
            .res = al,
        };
    }

    pub fn getSpan(self: *const Self, len: u8) SourceSpan {
        return SourceSpan{
            .line = self.row,
            .col = self.cur - self.bol + 1 - len,
            .len = len,
        };
    }

    pub fn newToken(self: *const Self, data: TokenData, len: u8) Token {
        return Token{ .data = data, .span = self.getSpan(len) };
    }

    pub fn newTokenVoid(self: *const Self, comptime kind: TokenData, len: u8) Token {
        return self.newToken(@unionInit(TokenData, @tagName(kind), {}), len);
    }

    pub fn deinit(self: *Self) void {
        self.keywords.deinit(self.alloc);
        self.types.deinit(self.alloc);
        // we do not deinit the res
    }

    const isWhitespace = std.ascii.isWhitespace;

    fn isSeparator(ch: u8) bool {
        // we don't need . because its not in the language
        return std.mem.count(u8, "{}[]():,", &[_]u8{ch}) > 0;
    }

    fn isOperatorStart(ch: u8) bool {
        return std.mem.count(u8, "+-*/=<>", &[_]u8{ch}) > 0;
    }

    fn isNumeral(potentialNum: []const u8) bool {
        if (potentialNum.len == 1 and potentialNum[0] == '-') {
            return false;
        }

        var slc = potentialNum;
        if (potentialNum[0] == '-') {
            if (!std.ascii.isDigit(potentialNum[1])) {
                return false;
            }
            slc = potentialNum[1..potentialNum.len];
        }

        for (slc) |ch| {
            if (!std.ascii.isDigit(ch) and ch != '_' and ch != '.') {
                return false;
            }
        }

        return true;
    }

    fn isCaseConsistent(s: []const u8) bool {
        return util.isLowercase(s) or util.isUppercase(s);
    }

    fn isKeyword(self: *const Self, word: []const u8) bool {
        if (!isCaseConsistent(word)) return false;
        const haystack = self.keywords.keyIterator().items;
        return std.mem.count([]const u8, haystack, word) > 0;
    }

    fn isType(self: *const Self, word: []const u8) bool {
        if (!isCaseConsistent(word)) return false;
        const haystack = self.types.keyIterator().items;
        return std.mem.count([]const u8, haystack, word) > 0;
    }

    fn trimLeft(self: *Lexer) void {
        if (self.cur >= self.file.len) {
            return;
        }

        while (self.cur < self.file.len and std.ascii.isWhitespace(self.file[self.cur]) and self.file[self.cur] != '\n')
            self.cur += 1;

        self.trimComment();
    }

    fn trimComment(self: *Lexer) void {
        if (self.cur + 2 > self.file.len) {
            return;
        }

        if (std.mem.eql(u8, self.file[self.cur .. self.cur + 2], "//")) {
            self.cur += 2;
            while (self.cur < self.file.len and self.file[self.cur] != '\n') {
                self.cur += 1;
            }
            self.cur += 1; // get rid of newline

            return self.trimLeft();
        }

        if (std.mem.eql(u8, self.file[self.cur .. self.cur + 2], "/*")) {
            self.cur += 2;
            // account for 2 characters
            while (self.cur < self.file.len and !std.mem.eql(u8, self.file[self.cur .. self.cur + 2], "*/")) {
                if (self.file[self.cur] == '\n') {
                    self.row += 1;
                    self.bol = self.cur + 1;
                }
                self.cur += 1;
            }
            // when we find */, we have to skip 2 past to avoid parsing it
            self.cur += 2;

            return self.trimLeft();
        }
    }

    pub fn nextOperator(self: *Lexer) ?Token {
        if (self.cur + 3 < self.file.len) {
            const curTriplet = self.file[self.cur .. self.cur + 3];

            var op: ?TokenData = null;
            if (std.mem.eql(u8, curTriplet, "**=")) {
                op = .pow_assign;
            }

            if (op) |res| {
                self.cur += 3;
                return Token{ .data = res, .span = self.getSpan(3) };
            }
        }

        if (self.cur + 2 < self.file.len) {
            const curPair = self.file[self.cur .. self.cur + 2];

            var op: ?TokenData = null;
            if (std.mem.eql(u8, curPair, "==")) {
                op = .equal;
            } else if (std.mem.eql(u8, curPair, "<=")) {
                op = .less_than_or_equal;
            } else if (std.mem.eql(u8, curPair, ">=")) {
                op = .greater_than_or_equal;
            } else if (std.mem.eql(u8, curPair, "!=")) {
                op = .not_equal;
            } else if (std.mem.eql(u8, curPair, "-=")) {
                op = .sub_assign;
            } else if (std.mem.eql(u8, curPair, "+=")) {
                op = .add_assign;
            } else if (std.mem.eql(u8, curPair, "*=")) {
                op = .mul_assign;
            } else if (std.mem.eql(u8, curPair, "/=")) {
                op = .div_assign;
            } else if (std.mem.eql(u8, curPair, "**")) {
                op = .pow;
            }

            if (op) |res| {
                self.cur += 2;
                return Token{ .data = res, .span = self.getSpan(2) };
            }
        }

        const operator: TokenData = switch (self.file[self.cur]) {
            '>' => .greater_than,
            '<' => .less_than,
            '=' => .assign,
            '*' => .mul,
            '/' => .div,
            '+' => .add,
            '-' => .sub,
            else => {
                return null;
            },
        };

        // operator is not null
        if (operator == .sub and std.ascii.isDigit(self.file[self.cur + 1]))
            return null; // force rescanning as a word

        self.cur += 1;
        return Token{ .data = operator, .span = self.getSpan(1) };
    }

    pub fn nextSeparator(self: *Lexer) ?Token {
        const sym: TokenData = switch (self.file[self.cur]) {
            '{' => .left_curly,
            '}' => .right_curly,
            '[' => .left_bracket,
            ']' => .right_bracket,
            '(' => .left_paren,
            ')' => .right_paren,
            ':' => .colon,
            ',' => .comma,
            else => {
                return null;
            },
        };
        self.cur += 1;
        return Token{ .data = sym, .span = self.getSpan(1) };
    }

    // this is an absoluteCinema™ function. do not question why this works. ths is absolute sorcery from december 2024. bruh.
    fn nextWord(self: *Lexer) []const u8 {
        const begin = self.cur;
        var currChar = self.file[self.cur];
        var end = self.cur;
        const stringOrCharLiteral = (currChar == '\'' or currChar == '"');
        const beginChar = currChar; // for the.primitive stuff

        // TODO: cleaner stuff with comptime?
        if (!stringOrCharLiteral) {
            var shouldSliceNow = false;
            if (currChar == '-') {
                // i love negative number
                self.cur += 1; // skip past -
                end += 1;
                currChar = self.file[self.cur];

                // sorcery from the python era
                const prevToken = self.res.getLastOrNull();
                if (prevToken) |tok| {
                    shouldSliceNow = switch (tok.data) {
                        .ident, .primitive => true,
                        .right_bracket, .right_paren, .right_curly => true,
                        else => false,
                    };
                }
            }

            // we increment cur by 1 in the loop so account for that (see last component of condition)
            if (!shouldSliceNow) {
                while (!isWhitespace(currChar) and !isSeparator(currChar) and !isOperatorStart(currChar)) {
                    // scan the word
                    self.cur += 1;
                    end += 1;

                    // post condition this shit cos its fucking sorcery
                    if (self.cur >= self.file.len) {
                        break;
                    }

                    currChar = self.file[self.cur];
                }
            }
        } else {
            // skip past the initial "
            self.cur += 1;
            end += 1;
            currChar = self.file[self.cur];
            // we increment cur by 1 in the loop so account for that (see last component of condition)
            while (currChar != beginChar) {
                self.cur += 1;
                end += 1;

                // post condition this shit cos its fucking sorcery
                if (self.cur >= self.file.len) {
                    break;
                }

                currChar = self.file[self.cur];
            }
            // account for ending quote
            self.cur += 1;
            end += 1;
        }

        if (end >= self.file.len) {
            const span = SourceSpan{
                .len = @truncate(self.file.len - begin),
                .line = self.row,
                .col = begin,
            };

            if (stringOrCharLiteral) {
                util.diag(&span, self.file_name, "unexpected end of file while scanning for string or character literal", .{});
            } else {
                util.diag(&span, self.file_name, "unexpected end of file while scanning for word", .{});
            }
            util.fatal("lexer", "compilation cannot continue.", .{});
        }

        const result = self.alloc.alloc(u8, end - begin) catch |err| util.panic(err);
        std.mem.copyForwards(u8, result, self.file[begin..end]);
        return result;
    }

    fn nextKeyword(self: *Lexer, word: []const u8) ?Token {
        if (!isCaseConsistent(word)) return null;

        const w_upper = std.ascii.allocUpperString(self.alloc, word) catch |err| util.panic(err);
        defer self.alloc.free(w_upper);

        if (self.keywords.get(w_upper)) |kw| {
            const span = SourceSpan{ .line = self.row, .col = @truncate(self.cur - self.bol - w_upper.len + 1), .len = @truncate(w_upper.len) };
            const tok = Token{ .data = kw, .span = span };
            return tok;
        }

        return null;
    }

    fn nextType(self: *Lexer, word: []const u8) ?Token {
        if (!isCaseConsistent(word)) return null;

        const w_upper = std.ascii.allocUpperString(self.alloc, word) catch |err| util.panic(err);
        defer self.alloc.free(w_upper);

        if (self.types.get(w_upper)) |typ| {
            const span = SourceSpan{ .line = self.row, .col = @truncate(self.cur - self.bol - w_upper.len + 1), .len = @truncate(w_upper.len) };
            const tok = Token{ .data = typ, .span = span };
            return tok;
        }

        return null;
    }

    fn nextStringOrCharLiteral(self: *Lexer, word: []const u8) ?Token {
        const slc = word[1 .. word.len - 1];

        // col is magical sorcery
        const span = SourceSpan{ .line = self.row, .col = @truncate(self.cur - self.bol - (slc.len + 1)), .len = @truncate(word.len) };

        if (word[0] == '"' and word[word.len - 1] == '"') {
            // include chars around
            const p = Primitive.init(.string, self.alloc, slc);
            return Token.init(.{ .primitive = p }, span);
        }

        if (word[0] == '\'' and word[word.len - 1] == '\'') {
            // we will check its validity later, to allow for escape sequences
            const p = Primitive.init(.char, self.alloc, slc);
            return Token.init(.{ .primitive = p }, span);
        }

        return null;
    }

    fn nextBoolean(self: *const Self, word: []const u8) ?Token {
        if (!isCaseConsistent(word)) {
            return null;
        }

        if (!std.ascii.eqlIgnoreCase("TRUE", word) and !std.ascii.eqlIgnoreCase("FALSE", word))
            return null;

        const span = SourceSpan{ .line = self.row, .col = @truncate(self.cur - word.len + 1), .len = @truncate(word.len) };
        const p = Primitive.init(.bool, self.alloc, word);
        return Token.init(.{ .primitive = p }, span);
    }

    pub fn nextToken(self: *Lexer) ?Token {
        // goodbye comment, goodbye whitespace
        self.trimLeft();

        if (self.cur >= self.file.len) {
            return null;
        }

        const currChar = self.file[self.cur];
        if (currChar == '\n') {
            self.row += 1;
            self.bol = self.cur + 1;
            self.cur += 1;
            return self.newTokenVoid(.newline, 1);
        }

        if (self.nextOperator()) |tok| return tok;
        if (self.nextSeparator()) |tok| return tok;

        const word = self.nextWord();

        if (self.res.items.len != 0) {
            // kill false positive case ?
            const last = self.res.getLast();
            const shouldBeSub = switch (last.data) {
                .ident, .primitive => true,
                .right_bracket, .right_paren, .right_curly => true,
                else => false,
            };

            if (word.len == 1 and shouldBeSub and word[0] == '-') {
                return self.newTokenVoid(.sub, 1);
            }
        }

        defer self.alloc.free(word);

        if (isNumeral(word)) {
            const p = Primitive.init(.number, self.alloc, word);
            return self.newToken(.{ .primitive = p }, @truncate(word.len));
        } else if (word[0] == '-' and !std.ascii.isDigit(word[1])) {
            // why is this even here :skull:
            // scuffed but even more scuffed cos i rewrote tis from the python version
            self.cur += 1;
            return self.newTokenVoid(.sub, 1);
        }

        if (self.nextKeyword(word)) |tok| return tok;
        if (self.nextType(word)) |tok| return tok;
        if (self.nextStringOrCharLiteral(word)) |tok| return tok;
        if (self.nextBoolean(word)) |tok| return tok;

        // return ident by default
        return Token.initIdent(self.alloc, word, self.getSpan(@truncate(word.len)));
    }

    pub fn tokenize(self: *Lexer) !std.ArrayListUnmanaged(Token) {
        // i hate putting res in self but oh well python sorcery
        while (self.cur < self.file.len) {
            if (self.nextToken()) |tok| {
                try self.res.append(self.alloc, tok);
            } else break;
        }

        try self.res.append(self.alloc, self.newTokenVoid(.eof, 1));

        return self.res;
    }
};
