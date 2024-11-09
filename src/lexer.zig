const std = @import("std");
const util = @import("./util.zig");

pub const Operator = enum { assign, equal, less_than, greater_than, less_than_or_equal, greater_than_or_equal, not_equal, mul, div, add, sub };

pub const Separator = enum {
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    left_curly,
    right_curly,
    colon,
    comma,
    dot,
};

// all prefixed with kw_ to avoid clashes
pub const Keyword = enum {
    kw_declare,
    kw_constant,
    kw_output,
    kw_input,
    kw_and,
    kw_or,
    kw_not,
    kw_if,
    kw_then,
    kw_else,
    kw_endif,
    kw_case,
    kw_of,
    kw_otherwise,
    kw_endcase,
    kw_while,
    kw_do,
    kw_endwhile,
    kw_repeat,
    kw_until,
    kw_for,
    kw_to,
    kw_next,
    kw_procedure,
    kw_endprocedure,
    kw_call,
    kw_function,
    kw_returns,
    kw_endfunction,
    kw_openfile,
    kw_readfile,
    kw_writefile,
    kw_closefile,
    kw_div,
    kw_mod,
};

pub const Type = enum {
    integer,
    real,
    boolean,
    string,
    char,
    array,
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
    identifier,
    literal,
    operator,
    separator,
    _type,
};

// All strings are owned slices on the heap
pub const Token = union(TokenKind) {
    keyword: Keyword,
    identifier: []const u8,
    literal: Literal,
    operator: Operator,
    separator: Separator,
    // undescore to avoid clash with zig compiler
    _type: Type,

    pub fn printToken(self: Token) void {
        switch (self) {
            .identifier => |ident| std.debug.print("token(ident): {s}\n", .{ident}),
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
        }
    }
};

pub const Tokens = struct {
    alloc: std.mem.Allocator,
    array_list: std.ArrayList(Token),

    pub fn init(alloc: std.mem.Allocator) !Tokens {
        return Tokens{
            .alloc = alloc,
            .array_list = std.ArrayList(Token).init(alloc),
        };
    }

    pub fn deinit(self: *Tokens) void {
        for (self.array_list.items) |token| {
            switch (token) {
                .keyword, .identifier, .literal => |kw| self.alloc.free(kw),
                else => continue,
            }
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

    pub fn init(alloc: std.mem.Allocator, file: []const u8) Lexer {
        var keywords = std.StringHashMap(Keyword).init(alloc);

        keywords.put("DECLARE", Keyword.kw_declare) catch |err| util.panic(err);
        keywords.put("CONSTANT", Keyword.kw_constant) catch |err| util.panic(err);
        keywords.put("OUTPUT", Keyword.kw_output) catch |err| util.panic(err);
        keywords.put("INPUT", Keyword.kw_input) catch |err| util.panic(err);
        keywords.put("AND", Keyword.kw_and) catch |err| util.panic(err);
        keywords.put("OR", Keyword.kw_or) catch |err| util.panic(err);
        keywords.put("NOT", Keyword.kw_not) catch |err| util.panic(err);
        keywords.put("IF", Keyword.kw_if) catch |err| util.panic(err);
        keywords.put("THEN", Keyword.kw_then) catch |err| util.panic(err);
        keywords.put("ENDIF", Keyword.kw_endif) catch |err| util.panic(err);
        keywords.put("CASE", Keyword.kw_case) catch |err| util.panic(err);
        keywords.put("OF", Keyword.kw_of) catch |err| util.panic(err);
        keywords.put("ENDCASE", Keyword.kw_endcase) catch |err| util.panic(err);
        keywords.put("WHILE", Keyword.kw_while) catch |err| util.panic(err);
        keywords.put("DO", Keyword.kw_do) catch |err| util.panic(err);
        keywords.put("ENDWHILE", Keyword.kw_endwhile) catch |err| util.panic(err);
        keywords.put("REPEAT", Keyword.kw_repeat) catch |err| util.panic(err);
        keywords.put("UNTIL", Keyword.kw_until) catch |err| util.panic(err);
        keywords.put("FOR", Keyword.kw_for) catch |err| util.panic(err);
        keywords.put("TO", Keyword.kw_to) catch |err| util.panic(err);
        keywords.put("NEXT", Keyword.kw_next) catch |err| util.panic(err);
        keywords.put("PROCEDURE", Keyword.kw_procedure) catch |err| util.panic(err);
        keywords.put("CALL", Keyword.kw_call) catch |err| util.panic(err);
        keywords.put("ENDPROCEDURE", Keyword.kw_endprocedure) catch |err| util.panic(err);
        keywords.put("CASE", Keyword.kw_case) catch |err| util.panic(err);
        keywords.put("FUNCTION", Keyword.kw_function) catch |err| util.panic(err);
        keywords.put("RETURNS", Keyword.kw_returns) catch |err| util.panic(err);
        keywords.put("ENDFUNCTION", Keyword.kw_endfunction) catch |err| util.panic(err);
        keywords.put("OPENFILE", Keyword.kw_openfile) catch |err| util.panic(err);
        keywords.put("CLOSEFILE", Keyword.kw_closefile) catch |err| util.panic(err);
        keywords.put("WRITEFILE", Keyword.kw_writefile) catch |err| util.panic(err);
        keywords.put("READFILE", Keyword.kw_readfile) catch |err| util.panic(err);
        keywords.put("DIV", Keyword.kw_div) catch |err| util.panic(err);
        keywords.put("MOD", Keyword.kw_mod) catch |err| util.panic(err);

        var types = std.StringHashMap(Type).init(alloc);

        types.put("INTEGER", Type.integer) catch |err| util.panic(err);
        types.put("REAL", Type.real) catch |err| util.panic(err);
        types.put("STRING", Type.string) catch |err| util.panic(err);
        types.put("CHAR", Type.char) catch |err| util.panic(err);
        types.put("BOOLEAN", Type.boolean) catch |err| util.panic(err);
        types.put("ARRAY", Type.array) catch |err| util.panic(err);

        return Lexer{
            .alloc = alloc,
            .file = file,
            .cur = 0,
            .bol = 0,
            .row = 1,
            .keywords = keywords,
            .types = types,
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.keywords.deinit();
        self.types.deinit();
    }

    const isWhitespace = std.ascii.isWhitespace;

    fn isSeparator(ch: u8) bool {
        return std.mem.count(u8, "{}[]():,.", &[_]u8{ch}) > 0;
    }

    fn isOperatorStart(ch: u8) bool {
        return std.mem.count(u8, "+-*/=<>", &[_]u8{ch}) > 0;
    }

    fn isNumeral(potentialNum: []const u8) bool {
        for (potentialNum) |ch| {
            if (!std.ascii.isDigit(ch) and ch != '_' and ch != '.') {
                return false;
            }
        }
        return true;
    }

    fn isKeyword(self: *Lexer, word: []const u8) bool {
        const haystack = self.keywords.keyIterator().items;
        std.mem.count([]const u8, haystack, word);
    }

    fn isType(self: *Lexer, word: []const u8) bool {
        const haystack = self.types.keyIterator().items;
        std.mem.count([]const u8, haystack, word);
    }

    fn trimLeft(self: *Lexer) void {
        if (self.cur >= self.file.len) {
            return;
        }

        while (self.cur < self.file.len and std.ascii.isWhitespace(self.file[self.cur])) {
            if (self.file[self.cur] == '\n') {
                self.row += 1;
                self.bol = self.cur + 1;
            }

            self.cur += 1;
        }

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

            self.trimLeft();
        }

        if (std.mem.eql(u8, self.file[self.cur .. self.cur + 2], "/*")) {
            self.cur += 2;
            // account for 2 characters
            while (self.cur + 1 < self.file.len and !std.mem.eql(u8, self.file[self.cur .. self.cur + 2], "*/")) {
                if (self.file[self.cur] == '\n') {
                    self.row += 1;
                    self.bol = self.cur + 1;
                }
                self.cur += 1;
            }
            // when we find */, we have to skip 2 past to avoid parsing it
            self.cur += 2;

            self.trimLeft();
        }
    }

    pub fn nextOperator(self: *Lexer) ?Token {
        if (self.cur + 2 < self.file.len) {
            const curPair = self.file[self.cur .. self.cur + 2];

            if (std.mem.eql(u8, curPair, "<-")) {
                self.cur += 2;
                return Token{ .operator = Operator.assign };
            } else if (std.mem.eql(u8, curPair, "<=")) {
                self.cur += 2;
                return Token{ .operator = Operator.less_than_or_equal };
            } else if (std.mem.eql(u8, curPair, ">=")) {
                self.cur += 2;
                return Token{ .operator = Operator.greater_than_or_equal };
            } else if (std.mem.eql(u8, curPair, "<>")) {
                self.cur += 2;
                return Token{ .operator = Operator.not_equal };
            }
        }

        var operator: Operator = undefined;
        operator = switch (self.file[self.cur]) {
            '>' => Operator.greater_than,
            '<' => Operator.less_than,
            '=' => Operator.equal,
            '*' => Operator.mul,
            '/' => Operator.div,
            '+' => Operator.add,
            '-' => Operator.sub,
            else => {
                return null;
            },
        };
        self.cur += 1;

        return Token{ .operator = operator };
    }

    pub fn nextSeparator(self: *Lexer) ?Token {
        const sym = switch (self.file[self.cur]) {
            '{' => Separator.left_curly,
            '}' => Separator.right_curly,
            '[' => Separator.left_bracket,
            ']' => Separator.right_bracket,
            '(' => Separator.left_paren,
            ')' => Separator.right_paren,
            ':' => Separator.colon,
            ',' => Separator.comma,
            '.' => Separator.dot,
            else => {
                return null;
            },
        };
        self.cur += 1;
        return Token{ .separator = sym };
    }

    fn nextWord(self: *Lexer) []const u8 {
        const begin = self.cur;
        var currChar = self.file[self.cur];
        var end = self.cur;
        const stringOrCharLiteral = (currChar == '\'' or currChar == '"');
        const beginChar = currChar; // for the literal stuff

        // TODO: cleaner stuff with comptime?
        if (!stringOrCharLiteral) {
            // we increment cur by 1 in the loop so account for that (see last component of condition)
            while (!isWhitespace(currChar) and !isSeparator(currChar) and !isOperatorStart(currChar)) {
                self.cur += 1;
                end += 1;

                // post condition this shit cos its fucking sorcery
                if (self.cur >= self.file.len) {
                    break;
                }

                currChar = self.file[self.cur];
            }
        } else {
            // skip past the initial "
            self.cur += 1;
            end += 1;
            currChar = self.file[self.cur];
            // we increment cur by 1 in the loop so account for that (see last component of condition)
            while (currChar != beginChar and !isSeparator(currChar) and !isOperatorStart(currChar)) {
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
        if (self.keywords.get(word)) |kw| {
            return Token{ .keyword = kw };
        }

        return null;
    }

    fn nextType(self: *Lexer, word: []const u8) ?Token {
        if (self.types.get(word)) |typ| {
            return Token{ ._type = typ };
        }

        return null;
    }

    fn nextStringOrCharLiteral(self: *Lexer, word: []const u8) ?Token {
        if (word[0] == '"' and word[word.len - 1] == '"') {
            const slc = word[1 .. word.len - 1];
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
        if (std.mem.eql(u8, "TRUE", word)) {
            return true;
        } else if (std.mem.eql(u8, "FALSE", word)) {
            return false;
        } else return null;
    }

    pub fn nextToken(self: *Lexer) ?Token {
        // goodbye comment, goodbye whitespace
        self.trimLeft();

        if (self.cur >= self.file.len) {
            return null;
        }

        if (self.nextOperator()) |tok| return tok;
        if (self.nextSeparator()) |tok| return tok;

        const word = self.nextWord();

        if (isNumeral(word)) {
            return Token{ .literal = Literal{ .number = word } };
        }

        if (self.nextKeyword(word)) |tok| return tok;
        if (self.nextType(word)) |tok| return tok;
        if (self.nextStringOrCharLiteral(word)) |tok| return tok;

        if (nextBoolean(word)) |b| {
            return Token{ .literal = Literal{ .boolean = b } };
        }

        // return ident by default
        return Token{ .identifier = word };
    }

    pub fn tokenize(self: *Lexer) !std.ArrayList(Token) {
        var res = std.ArrayList(Token).init(self.alloc);

        while (self.cur < self.file.len) {
            if (self.nextToken()) |tok| {
                try res.append(tok);
                tok.printToken();
            } else break;
        }

        return res;
    }
};
