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

    var lx = lexer.Lexer.init(alloc, content, file);
    defer lx.deinit();
    const tokens = try lx.tokenize();

    for (tokens.items) |item| {
        item.printToken();
    }

    defer tokens.deinit();
}

test "ast printing" {
    const alloc = std.heap.page_allocator;
    // TODO: get rid of this. forces compiler to check errs
    const printer = ast_printer.AstPrinter.init(alloc, std.io.getStdErr().writer().any());

    // print 69, "poopoo"
    const s_print = ast.PrintStmt{
        .items = &.{
            ast.Expr{ .e_literal = &ast.Literal{ .primitive = ast.Primitive{ .int = 69 } } },
            ast.Expr{ .e_literal = &ast.Literal{ .primitive = ast.Primitive{ .string = "poopoo" } } },
        },
    };

    // read somevar
    const s_read = ast.ReadStmt{ .ident = ast.Identifier{ .name = "somevar" } };

    // const Foo = 5
    const s_const = ast.ConstStmt{ .ident = "Foo", .value = ast.Expr{ .e_literal = &ast.Literal{ .primitive = ast.Primitive{ .int = 5 } } }, .exp = false };

    // var Bar: int
    const s_var = ast.VarStmt{ .ident = "Bar", .value = null, .typ = .{ .primitive = .int }, .exp = false };

    // bar = "baz"
    const s_assign = ast.AssignStmt{
        .ident = .{ .name = "Bar" },
        .value = .{ .e_literal = &ast.Literal{ .primitive = ast.Primitive{ .string = "baz" } } },
    };

    const stmts: [5]ast.Statement = .{
        .{ .s_print = s_print },
        .{ .s_read = s_read },
        .{ .s_const = s_const },
        .{ .s_var = s_var },
        .{ .s_assign = s_assign },
    };

    const prog = ast.Program{
        .stmts = &stmts,
    };

    printer.visitProgram(prog);
}
