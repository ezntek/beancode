const std = @import("std");
const lexer = @import("lexer.zig");
const ast = @import("ast_types.zig");

pub const Parser = struct {
    alloc: std.mem.Allocator,
    tokens: []lexer.Token,
    cur: u32,

    pub fn init(alloc: std.mem.Allocator, tokens: []lexer.Token) Parser {
        return Parser{
            .alloc = alloc,
            .tokens = tokens,
            .cur = 0,
        };
    }

    pub fn program(self: *const Parser) ast.Program {
        _ = self;
    }
};
