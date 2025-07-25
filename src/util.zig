const std = @import("std");
const common = @import("common.zig");

pub const Location = common.Location;

pub fn panic(err: anyerror) noreturn {
    std.debug.panic("the program encountered a fatal error: {any}", .{err});
}

pub fn fatal(location: ?[]const u8, comptime msg: []const u8, fmtargs: anytype) noreturn {
    if (location) |loc| {
        std.debug.print("\u{001b}[1;31merror(\u{001b}[0m{s}\u{001b}[1;31m)\u{001b}[0;2m: ", .{loc});
    } else {
        std.debug.print("\u{001b}[1;31merror\u{001b}[0;2m: ", .{});
    }

    std.debug.print(msg, fmtargs);

    if (location) |loc| {
        std.debug.print("\n\u{001b}[31m{s} STOP!\n\u{001b}[0m", .{loc});
    } else {
        std.debug.print("\n\u{001b}[1mcompile STOP!\n\u{001b}[0m", .{});
    }

    std.process.exit(1);
}

pub fn diag(loc: Location, comptime msg: []const u8, fmtargs: anytype) void {
    const stderr = std.io.getStdErr().writer();
    var bw = std.io.bufferedWriter(stderr);
    const writer = bw.writer();
    writer.print("{s}:{}:{}: ", .{ loc.file_name, loc.line, loc.col }) catch |err| panic(err);
    writer.print(msg, fmtargs) catch |err| panic(err);
    writer.writeByte('\n');
    bw.flush() catch |err| panic(err);
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
