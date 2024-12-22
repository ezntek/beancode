const std = @import("std");
const util = @import("./util.zig");
const lexer = @import("./lexer.zig");
const parser = @import("./parser.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const argv = std.os.argv;
    if (argv.len == 1) {
        util.fatal("cli", "gimme file gimme file", .{});
    }

    const file = std.mem.span(argv[1]);
    const fp = try std.fs.cwd().openFile(file, .{});
    defer fp.close();

    const content = try fp.readToEndAlloc(alloc, 262144);
    defer alloc.free(content);

    var lx = lexer.Lexer.init(alloc, content);
    defer lx.deinit();
    const tokens = try lx.tokenize();
    defer tokens.deinit();

    var ps = parser.Parser.init(alloc, tokens);
    const expr = ps.parse();

    std.debug.print("{any}\n", .{expr});
}
