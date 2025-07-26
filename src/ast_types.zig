// unlike the cursed ultra hyper absolute cinema pro max™ python impl,
// these types will only represent ast types.

const std = @import("std");
const util = @import("util.zig");
const printer = @import("ast_printer.zig");
const panic = util.panic;

pub const Operator = enum {
    assign,
    equal,
    less_than,
    greater_than,
    less_than_or_equal,
    greater_than_or_equal,
    not_equal,
    mul,
    div,
    add,
    sub,
    pow,
    mul_assign,
    div_assign,
    add_assign,
    sub_assign,
    pow_assign,
};

pub const Separator = enum {
    left_paren,
    right_paren,
    left_bracket,
    right_bracket,
    left_curly,
    right_curly,
    colon,
    comma,
    dot,
};

pub const PrimitiveType = enum {
    int,
    float,
    char,
    string,
    bool,
};

pub const ArrayType = struct {
    type: *const Type, // allow for nested arrays
    len: u32,
};

pub const Type = union(enum) {
    primitive: PrimitiveType,
    array: ArrayType,
};

pub const ArrayLiteral = struct {
    items: []Expr,
    len: Expr,
};

pub const Primitive = union(PrimitiveType) {
    int: i32,
    float: f64,
    char: u8,
    string: []const u8, // owned slice
    bool: bool,
};

pub const Typecast = struct {
    type: PrimitiveType,
    expr: Expr,
};

pub const BinaryExpr = struct {
    lhs: Expr,
    op: Operator,
    rhs: Expr,
};

pub const Literal = union(enum) {
    primitive: Primitive,
    array: ArrayLiteral,

    pub fn makePrimitive(v: Primitive) Literal {
        return Literal{ .primitive = v };
    }
};

pub const ArrayIndex = struct {
    expr: Expr,
    idx: u32,
};

pub const FunctionCall = struct {
    name: Expr, // just assume everything is callable
    args: []Expr,
};

// array indices that can work as idents in all statements such as assignments
pub const ArrayIndexIdentifier = struct {
    ident: *const Identifier,
    idx: u32,
};

pub const Identifier = union(enum) {
    name: []const u8,
    array_index: ArrayIndexIdentifier,
    function_call: FunctionCall,
};

pub const ExprKind = enum {
    e_literal, // union of ptrs
    e_negation,
    e_not,
    e_grouping,
    e_ident, // union of ptrs
    e_typecast,
    e_array_literal,
    e_array_index,
    e_array_index_identifier,
    e_function_call,
    e_binary,
};

pub const Expr = union(ExprKind) {
    e_literal: *const Literal, // union of ptrs
    e_negation: *const Expr,
    e_not: *const Expr,
    e_grouping: *const Expr,
    e_ident: *const Identifier, // union of ptrs
    e_typecast: *const Typecast,
    e_array_literal: *const ArrayLiteral,
    e_array_index: *const ArrayIndex,
    e_array_index_identifier: *const ArrayIndexIdentifier,
    e_function_call: *const FunctionCall,
    e_binary: *const BinaryExpr,

    pub fn make(comptime kind: ExprKind, alloc: std.mem.Allocator, v: anytype) Expr {
        const res = alloc.create(@TypeOf(v)) catch |err| panic(err);
        res.* = v;
        return @unionInit(Expr, @tagName(kind), res);
    }

    pub fn destroy(self: *const Expr, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .e_literal => |v| alloc.destroy(v),
            .e_negation => |v| alloc.destroy(v),
            .e_not => |v| alloc.destroy(v),
            .e_grouping => |v| alloc.destroy(v),
            .e_ident => |v| alloc.destroy(v),
            .e_typecast => |v| alloc.destroy(v),
            .e_array_literal => |v| alloc.destroy(v),
            .e_array_index => |v| alloc.destroy(v),
            .e_array_index_identifier => |v| alloc.destroy(v),
            .e_function_call => |v| alloc.destroy(v),
            .e_binary => |v| alloc.destroy(v),
        }
    }

    // useful utility methods
    pub fn makePrimitive(alloc: std.mem.Allocator, v: Primitive) Expr {
        return Expr.make(.e_literal, alloc, Literal.makePrimitive(v));
    }
};

pub const PrintStmt = struct {
    items: []const Expr, // owned
};

pub const ReadStmt = struct {
    ident: Identifier,
};

pub const ConstStmt = struct {
    ident: []const u8,
    value: Expr,
    exp: bool,
};

pub const VarStmt = struct {
    ident: []const u8,
    typ: ?Type,
    exp: bool,
    value: ?Expr,
};

pub const AssignStmt = struct {
    ident: Identifier,
    value: Expr,
};

pub const Program = struct {
    stmts: []const Statement, // owned
};

pub const Statement = union(enum) {
    s_print: PrintStmt,
    s_read: ReadStmt,
    s_const: ConstStmt,
    s_var: VarStmt,
    s_assign: AssignStmt,
    s_program: Program,
};

test "make expr" {
    const alloc = std.heap.page_allocator;
    const expr = Expr.make(.e_literal, alloc, Literal.makePrimitive(.{ .int = 234 }));
    const w = std.io.getStdErr().writer().any();
    var p = printer.AstPrinter.init(alloc, w);
    p.visitExpr(expr);
    try w.writeByte('\n');
    expr.destroy(alloc);
}
