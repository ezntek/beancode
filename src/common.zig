const std = @import("std");

pub const CompilerStage = enum {
    lexer,
    parser,
    // TODO: add more
};

pub const SourceSpan = struct {
    line: u32,
    col: u32,
    len: u16,
    // multiple files will exist later

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;
        try writer.print("{}:{} {}", .{ self.line, self.col, self.len });
    }
};
