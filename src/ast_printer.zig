const std = @import("std");
const ast = @import("ast_types.zig");
const lexer = @import("lexer.zig");

const util = @import("util.zig");

pub const AstPrinter = struct {
    alloc: std.mem.Allocator,
    writer: std.io.AnyWriter,

    const Self = AstPrinter;

    pub fn init(alloc: std.mem.Allocator, writer: std.io.AnyWriter) AstPrinter {
        return AstPrinter{ .alloc = alloc, .writer = writer };
    }

    fn write(self: *const Self, s: []const u8) void {
        _ = self.writer.write(s) catch |err| util.panic(err);
    }

    pub fn visitPrimitiveType(self: *const Self, typ: *const ast.BCPrimitiveType) void {
        self.write(@tagName(typ));
    }

    pub fn visitValue(self: *const Self, val: *const ast.BCValue) void {
        _ = switch (val.*) {
            .int => |i| self.writer.print("{}", .{i}),
            .float => |f| self.writer.print("{}", .{f}),
            .char => |c| self.writer.print("'{}'", .{c}),
            .string => |s| self.writer.print("\"{s}\"", .{s}),
            .bool => |b| self.writer.print("{}", .{b}),
        } catch |err| util.panic(err);
    }

    pub fn visitLiteral(self: *const Self, lit: *const ast.BCValue) void {
        self.visitValue(lit);
    }

    pub fn visitNegation(self: *const Self, expr: ast.Expr) void {
        self.write("negation(");
        self.visitExpr(expr);
        self.write(")");
    }

    pub fn visitNot(self: *const Self, expr: ast.Expr) void {
        self.write("not(");
        self.visitExpr(expr);
        self.write(")");
    }

    pub fn visitGrouping(self: *const Self, expr: ast.Expr) void {
        self.write("grouping(");
        self.visitExpr(expr);
        self.write(")");
    }

    pub fn visitIdent(self: *const Self, ident: []const u8) void {
        self.write(ident);
    }

    pub fn visitTypecast(self: *const Self, tc: *ast.Typecast) void {
        self.write("typecast(");
        self.visitPrimitiveType(tc.typ);
        self.write(", ");
        self.visitExpr(tc.expr);
    }

    pub fn visitArrayLiteral(self: *const Self, lit: *const ast.BCArrayLiteral) void {
        self.write("{ ");
        for (lit.items, 0..) |item, i| {
            self.visitExpr(item);
            if (i < lit.items.len - 1) {
                self.write(", ");
            }
        }
        self.write(" }");
    }

    pub fn visitBinaryExpr(self: *const Self, bin: *const ast.BinaryExpr) void {
        self.write(@tagName(bin.op));
        self.write("(");
        self.visitExpr(bin.lhs);
        self.write(", ");
        self.visitExpr(bin.rhs);
        self.write(")");
    }

    pub fn visitArrayIndex(self: *const Self, index: *const ast.ArrayIndex) void {
        self.write("arrayindex(");
        self.visitIdent(index.ident);
        self.write(", ");
        self.write(index.idx);
        self.write(")");
    }

    pub fn visitExpr(self: *const Self, expr: ast.Expr) void {
        switch (expr) {
            .e_literal => |lit| self.visitLiteral(lit),
            else => unreachable, // TODO: add others
        }
    }

    pub fn visitPrintStmt(self: *const Self, s_print: *const ast.PrintStmt) void {
        self.write("print(");
        for (s_print.items) |item| {
            self.visitExpr(item);
        }
        self.write(")");
    }

    pub fn visitStmt(self: *const Self, stmt: *const ast.Statement) void {
        switch (stmt.*) {
            .s_print => |*s_print| self.visitPrintStmt(s_print),
            else => unreachable, // TODO: add others
        }
    }

    pub fn visitProgram(self: *const Self, p: ast.Program) void {
        for (p.stmts) |*stmt| {
            self.visitStmt(stmt);
        }
    }
};
