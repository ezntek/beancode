const std = @import("std");
const util = @import("./util.zig");
const ast = @import("./ast_types.zig");

// all prefixed with kw_ to avoid clashes
pub const Keyword = enum {
    kw_var,
    kw_const,
    kw_print,
    kw_read,
    kw_and,
    kw_or,
    kw_not,
    kw_if,
    kw_then,
    kw_else,
    kw_end,
    kw_switch,
    kw_otherwise,
    kw_while,
    kw_do,
    kw_repeat,
    kw_until,
    kw_for,
    kw_to,
    kw_fn,
    kw_break,
    kw_continue,
};

pub const Type = enum {
    int,
    float,
    bool,
    string,
    char,
};

pub const LiteralKind = enum {
    number,
    string,
    char,
    boolean,
};

pub const Literal = union(LiteralKind) {
    number: []const u8,
    string: []const u8,
    char: u8,
    boolean: bool,
};

pub const TokenKind = enum {
    keyword,
    ident,
    literal,
    operator,
    separator,
    _type,
    newline,
};

// All strings are owned slices on the heap
pub const Token = union(TokenKind) {
    keyword: Keyword,
    ident: []const u8,
    literal: Literal,
    operator: ast.Operator,
    separator: ast.Separator,
    // undescore to avoid clash with zig compiler
    _type: Type,
    newline,

    pub fn printToken(self: Token) void {
        switch (self) {
            .ident => |ident| std.debug.print("token(ident): {s}\n", .{ident}),
            .keyword => |kw| std.debug.print("token(keyword): {any}\n", .{kw}),
            .literal => |lit| switch (lit) {
                .string => |str| std.debug.print("token(literal): \"{s}\"\n", .{str}),
                .number => |num| std.debug.print("token(literal): {s}\n", .{num}),
                .char => |thing| std.debug.print("token(literal): {c}\n", .{thing}),
                .boolean => |thing| std.debug.print("token(literal): {}\n", .{thing}),
            },
            .operator => |op| std.debug.print("token(operator): {any}\n", .{op}),
            .separator => |sep| std.debug.print("token(separator): {any}\n", .{sep}),
            ._type => |typ| std.debug.print("token(type): {any}\n", .{typ}),
            .newline => std.debug.print("token(newline)\n", .{}),
        }
    }
};

pub const Lexer = struct {
    alloc: std.mem.Allocator,
    file: []const u8,
    cur: u32, // current char
    bol: u32, // begin of line
    row: u32, // row
    keywords: std.StringHashMap(Keyword),
    types: std.StringHashMap(Type),
    res: std.ArrayList(Token),

    pub fn init(alloc: std.mem.Allocator, file: []const u8) Lexer {
        var keywords = std.StringHashMap(Keyword).init(alloc);

        keywords.put("VAR", Keyword.kw_var) catch |err| util.panic(err);
        keywords.put("CONST", Keyword.kw_const) catch |err| util.panic(err);
        keywords.put("PRINT", Keyword.kw_print) catch |err| util.panic(err);
        keywords.put("READ", Keyword.kw_read) catch |err| util.panic(err);
        keywords.put("AND", Keyword.kw_and) catch |err| util.panic(err);
        keywords.put("OR", Keyword.kw_or) catch |err| util.panic(err);
        keywords.put("NOT", Keyword.kw_not) catch |err| util.panic(err);
        keywords.put("IF", Keyword.kw_if) catch |err| util.panic(err);
        keywords.put("THEN", Keyword.kw_then) catch |err| util.panic(err);
        keywords.put("ELSE", Keyword.kw_else) catch |err| util.panic(err);
        keywords.put("END", Keyword.kw_end) catch |err| util.panic(err);
        keywords.put("SWITCH", Keyword.kw_switch) catch |err| util.panic(err);
        keywords.put("WHILE", Keyword.kw_while) catch |err| util.panic(err);
        keywords.put("DO", Keyword.kw_do) catch |err| util.panic(err);
        keywords.put("REPEAT", Keyword.kw_repeat) catch |err| util.panic(err);
        keywords.put("UNTIL", Keyword.kw_until) catch |err| util.panic(err);
        keywords.put("FOR", Keyword.kw_for) catch |err| util.panic(err);
        keywords.put("TO", Keyword.kw_to) catch |err| util.panic(err);
        keywords.put("CASE", Keyword.kw_switch) catch |err| util.panic(err);
        keywords.put("FN", Keyword.kw_fn) catch |err| util.panic(err);
        keywords.put("BREAK", Keyword.kw_break) catch |err| util.panic(err);
        keywords.put("CONTINUE", Keyword.kw_continue) catch |err| util.panic(err);

        var types = std.StringHashMap(Type).init(alloc);

        types.put("INT", Type.int) catch |err| util.panic(err);
        types.put("FLOAT", Type.float) catch |err| util.panic(err);
        types.put("STRING", Type.string) catch |err| util.panic(err);
        types.put("CHAR", Type.char) catch |err| util.panic(err);
        types.put("BOOLEAN", Type.bool) catch |err| util.panic(err);

        return Lexer{
            .alloc = alloc,
            .file = file,
            .cur = 0,
            .bol = 0,
            .row = 1,
            .keywords = keywords,
            .types = types,
            .res = std.ArrayList(Token).init(alloc),
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.keywords.deinit();
        self.types.deinit();
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

    fn isKeyword(self: *const Lexer, word: []const u8) bool {
        if (!isCaseConsistent(word)) return false;
        const haystack = self.keywords.keyIterator().items;
        return std.mem.count([]const u8, haystack, word) > 0;
    }

    fn isType(self: *const Lexer, word: []const u8) bool {
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
        if (self.cur + 2 < self.file.len) {
            const curPair = self.file[self.cur .. self.cur + 2];

            if (std.mem.eql(u8, curPair, "==")) {
                self.cur += 2;
                return Token{ .operator = .equal };
            } else if (std.mem.eql(u8, curPair, "<=")) {
                self.cur += 2;
                return Token{ .operator = .less_than_or_equal };
            } else if (std.mem.eql(u8, curPair, ">=")) {
                self.cur += 2;
                return Token{ .operator = .greater_than_or_equal };
            } else if (std.mem.eql(u8, curPair, "!=")) {
                self.cur += 2;
                return Token{ .operator = .not_equal };
            }
        }

        const operator = switch (self.file[self.cur]) {
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
        return Token{ .operator = operator };
    }

    pub fn nextSeparator(self: *Lexer) ?Token {
        const sym = switch (self.file[self.cur]) {
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
        return Token{ .separator = sym };
    }

    // this is an absoluteCinema™ function. do not question why this works. ths is absolute sorcery from december 2024. bruh.
    fn nextWord(self: *Lexer) []const u8 {
        const begin = self.cur;
        var currChar = self.file[self.cur];
        var end = self.cur;
        const stringOrCharLiteral = (currChar == '\'' or currChar == '"');
        const beginChar = currChar; // for the literal stuff

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
                    shouldSliceNow = switch (tok) {
                        .ident, .literal => true,
                        .separator => |sep| switch (sep) {
                            .right_bracket, .right_paren, .right_curly => true,
                            else => false,
                        },
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

        const result = self.alloc.alloc(u8, end - begin) catch |err| util.panic(err);
        std.mem.copyForwards(u8, result, self.file[begin..end]);
        return result;
    }

    fn nextKeyword(self: *Lexer, word: []const u8) ?Token {
        if (!isCaseConsistent(word)) return null;

        const w_upper = std.ascii.allocUpperString(self.alloc, word) catch |err| util.panic(err);
        defer self.alloc.free(w_upper);

        if (self.keywords.get(w_upper)) |kw| {
            return Token{ .keyword = kw };
        }

        return null;
    }

    fn nextType(self: *Lexer, word: []const u8) ?Token {
        if (!isCaseConsistent(word)) return null;

        const w_upper = std.ascii.allocUpperString(self.alloc, word) catch |err| util.panic(err);
        defer self.alloc.free(w_upper);

        if (self.types.get(w_upper)) |typ| {
            return Token{ ._type = typ };
        }

        return null;
    }

    fn nextStringOrCharLiteral(self: *Lexer, word: []const u8) ?Token {
        const slc = word[1 .. word.len - 1];
        //const pos = ()
        if (word[0] == '"' and word[word.len - 1] == '"') {
            return Token{ .literal = Literal{ .string = slc } };
        }

        if (word[0] == '\'' and word[word.len - 1] == '\'') {
            if (word.len > 3) {
                const row = self.row;
                const col = self.cur - self.bol - word.len;
                util.fatal("lexer", "char literal at {},{} cannot contain more than 1 char!", .{ row, col });
            }

            return Token{ .literal = Literal{ .char = word[1] } };
        }

        return null;
    }

    fn nextBoolean(word: []const u8) ?bool {
        if (!isCaseConsistent(word)) {
            return null;
        }

        if (std.ascii.eqlIgnoreCase("TRUE", word)) {
            return true;
        } else if (std.ascii.eqlIgnoreCase("FALSE", word)) {
            return false;
        } else return null;
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
            return .newline;
        }

        if (self.nextOperator()) |tok| return tok;
        if (self.nextSeparator()) |tok| return tok;

        const word = self.nextWord();

        if (self.res.items.len != 0) {
            // kill false positive case ?
            const last = self.res.getLast();
            const shouldBeSub = switch (last) {
                .ident, .literal => true,
                .separator => |sep| switch (sep) {
                    .right_bracket, .right_paren, .right_curly => true,
                    else => false,
                },
                else => false,
            };
            if (word.len == 1 and shouldBeSub and word[0] == '-') {
                return Token{ .operator = .sub };
            }
        }

        if (isNumeral(word)) {
            return Token{ .literal = Literal{ .number = word } };
        } else if (word[0] == '-' and !std.ascii.isDigit(word[1])) {
            // why is this even here :skull:
            // scuffed but even more scuffed cos i rewrote tis from the python version
            self.cur += 1;
            return Token{ .operator = .sub };
        }

        if (self.nextKeyword(word)) |tok| return tok;
        if (self.nextType(word)) |tok| return tok;
        if (self.nextStringOrCharLiteral(word)) |tok| return tok;

        if (nextBoolean(word)) |b| {
            return Token{ .literal = Literal{ .boolean = b } };
        }

        // return ident by default
        return Token{ .ident = word };
    }

    pub fn tokenize(self: *Lexer) !std.ArrayList(Token) {
        // i hate putting res in self but oh well python sorcery
        while (self.cur < self.file.len) {
            if (self.nextToken()) |tok| {
                try self.res.append(tok);
            } else break;
        }

        return self.res;
    }
};
