const std = @import("std");

pub fn panic(err: anyerror) noreturn {
    std.debug.panic("the program encountered a fatal error: {any}", .{err});
}

pub fn fatal(location: []const u8, comptime msg: []const u8, fmtargs: anytype) noreturn {
    if (location.len != 0) {
        std.debug.print("\u{001b}[1;31merror(\u{001b}[0m{s}\u{001b}[1;31m)\u{001b}[0;2m: ", .{location});
    } else {
        std.debug.print("\u{001b}[1;31merror\u{001b}[0;2m: ", .{});
    }

    std.debug.print(msg, fmtargs);
    if (location.len != 0) {
        std.debug.print("\n\u{001b}[31m{s} STOP!\n\u{001b}[0m", .{location});
    } else {
        std.debug.print("\n\u{001b}[1mcompile STOP!\n\u{001b}[0m", .{});
    }

    std.process.exit(1);
}
