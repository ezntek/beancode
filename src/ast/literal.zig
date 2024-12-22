const std = @import("std");

pub const LiteralKind = enum {
    string,
    boolean,
    integer,
    real,
    char,
};

pub const Literal = union(LiteralKind) {
    string: []const u8, // ideallly heap-allocated
    boolean: bool,
    integer: i32, // allow stupid overflows and such
    real: f64,
    char: u8,
};
