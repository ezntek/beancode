const std = @import("std");
const util = @import("./util.zig");
const lexer = @import("./lexer.zig");

const ast = @import("./ast_types.zig");
const ast_printer = @import("./ast_printer.zig");

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

    for (tokens.items) |item| {
        item.printToken();
    }

    // TODO: get rid of this. forces compiler to check errs
    const printer = ast_printer.AstPrinter.init(alloc, std.io.getStdErr().writer().any());
    printer.visitProgram(ast.Program{ .stmts = &.{} });

    defer tokens.deinit();
}
