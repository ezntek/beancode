const std = @import("std");

const LiteralKind = enum {
    string,
    boolean,
    integer,
    real,
    char,
};

const Literal = union(LiteralKind) {
    string: []const u8, // ideallly heap-allocated
    boolean: bool,
    integer: i32, // allow stupid overflows and such
    real: f64,
    char: u8,
};
