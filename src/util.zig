const std = @import("std");
const common = @import("common.zig");

pub const SourceSpan = common.SourceSpan;

pub fn panic(err: anyerror) noreturn {
    std.debug.panic("the program encountered a fatal error: {any}", .{err});
}

pub fn fatal(location: ?[]const u8, comptime msg: []const u8, fmtargs: anytype) noreturn {
    const stderr = std.io.getStdErr().writer();
    var bw = std.io.bufferedWriter(stderr);
    const writer = bw.writer();

    writer.print("\u{1b}[1;31mfatal:\u{1b}[0m ", .{}) catch |err| panic(err);
    writer.print(msg, fmtargs) catch |err| panic(err);

    if (location) |loc| {
        writer.print("\n\u{1b}[1m{s} stop!\n\u{001b}[0m", .{loc}) catch |err| panic(err);
    } else {
        writer.print("\n\u{1b}[1mcompile stop!\n\u{001b}[0m", .{}) catch |err| panic(err);
    }

    bw.flush() catch |err| panic(err);
    std.process.exit(1);
}

pub fn diag(loc: *const SourceSpan, file_name: []const u8, comptime msg: []const u8, fmtargs: anytype) void {
    const stderr = std.io.getStdErr().writer();
    var bw = std.io.bufferedWriter(stderr);
    const writer = bw.writer();
    defer {
        bw.flush() catch |err| panic(err);
    }

    writer.print("\u{1b}[31;1merror: \u{1b}[0;1m{s}:{}:{}:\u{1b}[0m ", .{ file_name, loc.line, loc.col }) catch |err| panic(err);
    writer.print(msg, fmtargs) catch |err| panic(err);
    writer.writeByte('\n') catch |err| panic(err);
}

pub fn isUppercase(s: []const u8) bool {
    for (s) |c| {
        if (std.ascii.isLower(c))
            return false;
    }

    return true;
}

pub fn isLowercase(s: []const u8) bool {
    for (s) |c| {
        if (std.ascii.isUpper(c))
            return false;
    }

    return true;
}
