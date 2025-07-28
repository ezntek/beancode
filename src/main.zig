const std = @import("std");
const util = @import("util.zig");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");

const ast = @import("ast_types.zig");
const ast_printer = @import("ast_printer.zig");

pub fn main() !void {
    var dba = std.heap.DebugAllocator(.{}){};
    const alloc = dba.allocator();
    defer {
        _ = dba.deinit();
    }

    const argv = std.os.argv;
    if (argv.len == 1) {
        util.fatal("cli", "gimme file gimme file", .{});
    }

    const file = std.mem.span(argv[1]);
    const fp = try std.fs.cwd().openFile(file, .{});
    defer fp.close();

    const content = try fp.readToEndAlloc(alloc, 262144);
    defer alloc.free(content);

    var lx = lexer.Lexer.init(alloc, content, file);
    defer lx.deinit();
    var tokens = try lx.tokenize();
    defer {
        for (tokens.items) |tok| {
            tok.deinit(alloc);
        }
        tokens.deinit(alloc);
    }

    std.debug.print("\u{001b}[1m===== beginning of token list =====\u{001b}[0m\n", .{});

    for (tokens.items) |item| {
        std.debug.print("{s} {s}\n", .{ item, item.span });
    }

    var p = parser.Parser.init(alloc, file, tokens.items);
    defer p.deinit(alloc);

    const program = p.program();
    defer program.deinit(alloc);

    std.debug.print("\u{001b}[1m===== beginning of ast =====\u{001b}[0m\n", .{});

    const printer = ast_printer.AstPrinter.init(alloc, std.io.getStdErr().writer().any());
    printer.visitProgram(&program);
}
