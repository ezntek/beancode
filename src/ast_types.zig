// unlike the cursed ultra hyper absolute cinema pro max™ python impl,
// these types will only represent ast types.

const std = @import("std");
const util = @import("util.zig");
const printer = @import("ast_printer.zig");
const SourceSpan = @import("common.zig").SourceSpan;

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

pub const LvalueArrayIndex = struct {
    ident: *const Lvalue,
    idx: u32,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, ident: *const Lvalue, idx: u32) Self {
        const mem = alloc.create(Lvalue) catch |err| panic(err);
        mem.* = ident;
        return LvalueArrayIndex{ .ident = mem, .idx = idx };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) Self {
        self.ident.destroy(alloc);
        alloc.destroy(self.ident);
    }
};

// values that work as the left hand side of assignments and definitely
// have a location in memory
pub const Lvalue = union(enum) {
    ident: []const u8,
    array_index: LvalueArrayIndex,

    const Self = @This();

    pub fn initIdent(alloc: std.mem.Allocator, ident: []const u8) Self {
        const new_ident = alloc.dupe(u8, ident) catch |err| panic(err);
        return Lvalue{ .ident = new_ident };
    }

    pub fn initArrayIndex(array_index: LvalueArrayIndex) Self {
        return Lvalue{ .array_index = array_index };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) Self {
        switch (self.*) {
            .ident => |id| alloc.free(id),
            .array_index => |arridx| arridx.deinit(alloc),
        }
    }
};

pub const ExprKind = enum {
    e_literal, // union of ptrs
    e_negation,
    e_not,
    e_grouping,
    e_lvalue, // union of ptrs
    e_typecast,
    e_array_literal,
    e_array_index,
    e_array_index_identifier,
    e_function_call,
    e_binary,
};

pub const ExprData = union(ExprKind) {
    e_literal: *const Literal, // union of ptrs
    e_negation: *const Expr,
    e_not: *const Expr,
    e_grouping: *const Expr,
    e_lvalue: *const Lvalue, // union of ptrs
    e_typecast: *const Typecast,
    e_array_literal: *const ArrayLiteral,
    e_array_index: *const ArrayIndex,
    e_array_index_identifier: *const LvalueArrayIndex,
    e_function_call: *const FunctionCall,
    e_binary: *const BinaryExpr,

    pub fn init(comptime kind: ExprKind, alloc: std.mem.Allocator, v: anytype) ExprData {
        const res = alloc.create(@TypeOf(v)) catch |err| panic(err);
        res.* = v;
        return @unionInit(ExprData, @tagName(kind), res);
    }

    pub fn deinit(self: *const ExprData, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .e_literal => |v| alloc.destroy(v),
            .e_negation => |v| alloc.destroy(v),
            .e_not => |v| alloc.destroy(v),
            .e_grouping => |v| alloc.destroy(v),
            .e_lvalue => |v| alloc.destroy(v),
            .e_typecast => |v| alloc.destroy(v),
            .e_array_literal => |v| alloc.destroy(v),
            .e_array_index => |v| alloc.destroy(v),
            .e_array_index_identifier => |v| alloc.destroy(v),
            .e_function_call => |v| alloc.destroy(v),
            .e_binary => |v| alloc.destroy(v),
        }
    }
};

pub const Expr = struct {
    span: *const SourceSpan,
    data: ExprData,

    pub fn init(comptime kind: ExprKind, alloc: std.mem.Allocator, v: anytype, loc: SourceSpan) Expr {
        const data = ExprData.init(kind, alloc, v);
        return Expr{ .span = loc, .data = data };
    }

    pub fn destroy(self: *const Expr, alloc: std.mem.Allocator) void {
        return self.data.deinit(alloc);
    }

    // useful utility methods
    pub fn initPrimitive(alloc: std.mem.Allocator, v: Primitive, span: *const SourceSpan) Expr {
        const data = ExprData.init(.e_literal, alloc, Literal.makePrimitive(v));
        return Expr{ .span = span, .data = data };
    }

    pub fn initIdent(alloc: std.mem.Allocator, v: []const u8, span: *const SourceSpan) Expr {
        const newv = alloc.dupe(u8, v) catch |err| panic(err);
        const data = ExprData.init(.e_lvalue, alloc, Lvalue{ .ident = newv });
        return Expr{ .span = span, .data = data };
    }
};

pub const PrintStmt = struct {
    items: []const Expr, // owned

    const Self = @This();

    pub fn init(items: []const Expr, alloc: std.mem.Allocator) Self {
        const new_items = alloc.dupe(Expr, items) catch |err| panic(err);
        return PrintStmt{ .items = new_items };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        alloc.free(self.items);
    }
};

pub const ReadStmt = struct {
    ident: Lvalue,

    const Self = @This();

    pub fn init() Self {
        unreachable; // TODO: implement
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        _ = self;
        _ = alloc;
        unreachable; // TODO: implement
    }
};

pub const ConstStmt = struct {
    ident: []const u8,
    value: Expr,
    exp: bool,

    const Self = @This();

    pub fn init() Self {
        unreachable; // TODO: implement
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        _ = self;
        _ = alloc;
        unreachable; // TODO: implement
    }
};

pub const VarStmt = struct {
    ident: []const u8,
    typ: ?Type,
    exp: bool,
    value: ?Expr,

    const Self = @This();

    pub fn init() Self {
        unreachable; // TODO: implement
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        _ = self;
        _ = alloc;
        unreachable; // TODO: implement
    }
};

pub const AssignStmt = struct {
    ident: Lvalue,
    value: Expr,

    const Self = @This();

    pub fn init() Self {
        unreachable; // TODO: implement
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        _ = self;
        _ = alloc;
        unreachable; // TODO: implement
    }
};

pub const Program = struct {
    stmts: []const Statement, // owned

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, stmts: []const Statement) Self {
        const new_stmts = alloc.dupe(Statement, stmts) catch |err| panic(err);
        return Program{
            .stmts = new_stmts,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        for (self.stmts) |stmt| {
            stmt.deinit(alloc);
        }
    }
};

pub const StatementKind = enum {
    s_print,
    s_read,
    s_const,
    s_var,
    s_assign,
    s_program,
};

pub const StatementData = union(StatementKind) {
    s_print: PrintStmt,
    s_read: ReadStmt,
    s_const: ConstStmt,
    s_var: VarStmt,
    s_assign: AssignStmt,
    s_program: Program,

    const Self = @This();

    pub fn init(comptime kind: StatementKind, alloc: std.mem.Allocator, v: anytype) StatementData {
        const res = alloc.create(@TypeOf(v)) catch |err| panic(err);
        res.* = v;
        return @unionInit(ExprData, @tagName(kind), res);
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .s_print => |v| v.deinit(alloc),
            .s_read => |v| v.deinit(alloc),
            .s_const => |v| v.deinit(alloc),
            .s_var => |v| v.deinit(alloc),
            .s_assign => |v| v.deinit(alloc),
            .s_program => |v| v.deinit(alloc),
        }
    }
};

pub const Statement = struct {
    data: StatementData,
    span: *const SourceSpan,

    const Self = @This();

    pub fn init(comptime kind: StatementKind, alloc: std.mem.Allocator, v: anytype, span: *const SourceSpan) Statement {
        const res = StatementData.init(kind, alloc, v);
        return Statement{ .data = res, .span = span };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.data.deinit(alloc);
    }

    pub fn initPrintStmt(items: []const Expr, alloc: std.mem.Allocator, span: *const SourceSpan) Statement {
        const res = PrintStmt.init(items, alloc);
        return Statement{ .data = .{ .s_print = res }, .span = span };
    }
};

test "make expr" {
    const alloc = std.heap.page_allocator;
    const expr = ExprData.init(.e_literal, alloc, Literal.makePrimitive(.{ .int = 234 }));
    const w = std.io.getStdErr().writer().any();
    var p = printer.AstPrinter.init(alloc, w);
    p.visitExpr(expr);
    try w.writeByte('\n');
    expr.deinit(alloc);
}
