const std = @import("std");
const ast = @import("ast_types.zig");
const lexer = @import("lexer.zig");

const util = @import("util.zig");

// XXX: this is top tier scuffed shitcode.
// do not judge even i dont like this
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

    pub fn visitPrimitiveType(self: *const Self, typ: *const ast.PrimitiveType) void {
        self.write(@tagName(typ.*));
    }

    pub fn visitArrayType(self: *const Self, arr: *const ast.ArrayType) void {
        const len = arr.len;
        const typ = self.visitType(arr.type); // rekurshun™
        self.writer.print("[{}]{}", .{ len, typ }) catch |err| util.panic(err);
    }

    pub fn visitType(self: *const Self, typ: *const ast.Type) void {
        switch (typ.*) {
            .primitive => |*prim| self.visitPrimitiveType(prim),
            .array => |*arr| self.visitArrayType(arr),
        }
    }

    pub fn visitValue(self: *const Self, val: *const ast.Primitive) void {
        _ = switch (val.*) {
            .int => |i| self.writer.print("{}", .{i}),
            .float => |f| self.writer.print("{}", .{f}),
            .char => |c| self.writer.print("'{}'", .{c}),
            .string => |s| self.writer.print("\"{s}\"", .{s}),
            .bool => |b| self.writer.print("{}", .{b}),
        } catch |err| util.panic(err);
    }

    pub fn visitLiteral(self: *const Self, lit: *const ast.Literal) void {
        switch (lit.*) {
            .primitive => |*prim| self.visitValue(prim),
            .array => |*array| self.visitArrayLiteral(array),
        }
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

    pub fn visitIdent(self: *const Self, ident: *const ast.Identifier) void {
        self.write("ident(");
        switch (ident.*) {
            .name => |s| self.write(s),
            .array_index => |*arridx| self.visitArrayIndexIdentifier(arridx),
            .function_call => |*fncall| self.visitFunctionCall(fncall),
        }
        self.write(")");
    }

    pub fn visitTypecast(self: *const Self, tc: *const ast.Typecast) void {
        self.write("typecast(");
        self.visitPrimitiveType(&tc.type);
        self.write(", ");
        self.visitExpr(tc.expr);
    }

    pub fn visitArrayLiteral(self: *const Self, lit: *const ast.ArrayLiteral) void {
        self.write("{ ");
        for (lit.items, 0..) |item, i| {
            self.visitExpr(item);
            if (i < lit.items.len - 1) {
                self.write(", ");
            }
        }
        self.write(" }");
    }

    pub fn visitFunctionCall(self: *const Self, fncall: *const ast.FunctionCall) void {
        self.write("fncall(");
        self.visitExpr(fncall.name);
        self.write(", ");
        for (fncall.args, 0..) |exp, i| {
            self.visitExpr(exp);
            if (i < fncall.args.len - 1) {
                self.write(", ");
            }
        }
        self.write(")");
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
        self.visitExpr(index.expr);
        self.write(", ");
        self.writer.print("{}", .{index.idx}) catch |err| util.panic(err);
        self.write(")");
    }

    pub fn visitArrayIndexIdentifier(self: *const Self, index: *const ast.ArrayIndexIdentifier) void {
        self.write("arrayindex(");
        self.visitIdent(index.ident);
        self.write(", ");
        self.writer.print("{}", .{index.idx}) catch |err| util.panic(err);
        self.write(")");
    }

    pub fn visitExpr(self: *const Self, expr: ast.Expr) void {
        switch (expr) {
            .e_literal => |lit| self.visitLiteral(lit),
            .e_negation => |inner| self.visitNegation(inner.*),
            .e_not => |inner| self.visitNot(inner.*),
            .e_grouping => |inner| self.visitGrouping(inner.*),
            .e_ident => |id| self.visitIdent(id),
            .e_typecast => |tc| self.visitTypecast(tc),
            .e_array_literal => |arrlit| self.visitArrayLiteral(arrlit),
            .e_array_index => |arridx| self.visitArrayIndex(arridx),
            .e_array_index_identifier => |arridx| self.visitArrayIndexIdentifier(arridx),
            .e_function_call => |fncall| self.visitFunctionCall(fncall),
            .e_binary => |bin| self.visitBinaryExpr(bin),
        }
    }

    pub fn visitPrintStmt(self: *const Self, s_print: *const ast.PrintStmt) void {
        self.write("print(");
        for (s_print.items, 0..) |item, i| {
            self.visitExpr(item);
            if (i < s_print.items.len - 1) {
                self.write(", ");
            }
        }
        self.write(")");
    }

    pub fn visitReadStmt(self: *const Self, s_read: *const ast.ReadStmt) void {
        self.write("read(");
        self.visitIdent(&s_read.ident);
        self.write(")");
    }

    pub fn visitConstStmt(self: *const Self, s_const: *const ast.ConstStmt) void {
        // ident, value, export
        if (s_const.exp) {
            self.write("export_");
        }

        self.write("const(");
        self.visitIdent(&.{ .name = s_const.ident });
        self.write(", ");
        self.visitExpr(s_const.value);
        self.write(")");
    }

    pub fn visitVarStmt(self: *const Self, s_var: *const ast.VarStmt) void {
        // ident, ?type, ?value, export
        if (s_var.exp) {
            self.write("export_");
        }

        self.write("var(");
        self.visitIdent(&.{ .name = s_var.ident });

        if (s_var.typ) |typ| {
            self.write(", ");
            self.visitType(&typ);
        }

        if (s_var.value) |exp| {
            self.write(", ");
            self.visitExpr(exp);
        }

        self.write(")");
    }

    pub fn visitAssignStmt(self: *const Self, s_assign: *const ast.AssignStmt) void {
        // ident, value
        self.write("assign(");
        self.visitIdent(&s_assign.ident);
        self.write(", ");
        self.visitExpr(s_assign.value);
        self.write(")");
    }

    pub fn visitStmt(self: *const Self, stmt: *const ast.Statement) void {
        switch (stmt.*) {
            .s_print => |*s_print| self.visitPrintStmt(s_print),
            .s_read => |*s_read| self.visitReadStmt(s_read),
            .s_const => |*s_const| self.visitConstStmt(s_const),
            .s_var => |*s_var| self.visitVarStmt(s_var),
            .s_assign => |*s_assign| self.visitAssignStmt(s_assign),
            .s_program => |s_program| self.visitProgram(s_program),
        }
    }

    pub fn visitProgram(self: *const Self, p: ast.Program) void {
        for (p.stmts) |*stmt| {
            self.visitStmt(stmt);
            self.write("\n");
        }
    }
};
