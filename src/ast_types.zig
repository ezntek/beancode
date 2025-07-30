// unlike the cursed ultra hyper absolute cinema pro max™ python impl,
// these types will only represent ast types.

const std = @import("std");
const util = @import("util.zig");
const printer = @import("ast_printer.zig");
const SourceSpan = @import("common.zig").SourceSpan;
const l = @import("lexer.zig");

const panic = util.panic;

// NOTE: since Expr is a union of pointers, no indirection is needed

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
    o_and,
    o_or,

    const Self = @This();

    pub fn fromToken(data: l.TokenData) ?Self {
        return switch (data) {
            .assign => .assign,
            .equal => .equal,
            .less_than => .less_than,
            .greater_than => .greater_than,
            .less_than_or_equal => .less_than_or_equal,
            .greater_than_or_equal => .greater_than_or_equal,
            .not_equal => .not_equal,
            .mul => .mul,
            .div => .div,
            .add => .add,
            .sub => .sub,
            .pow => .pow,
            .mul_assign => .mul_assign,
            .div_assign => .div_assign,
            .add_assign => .add_assign,
            .sub_assign => .sub_assign,
            .pow_assign => .pow_assign,
            .k_or => .o_or,
            .k_and => .o_and,
            else => null,
        };
    }
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

    const Self = @This();

    pub fn fromToken(data: l.TokenData) ?Self {
        return switch (data) {
            .left_paren => .left_paren,
            .right_paren => .right_paren,
            .left_bracket => .left_bracket,
            .right_bracket => .right_bracket,
            .left_curly => .left_curly,
            .right_curly => .right_curly,
            .colon => .colon,
            .comma => .comma,
            .dot => .dot,
            else => null,
        };
    }
};

pub const PrimitiveType = enum {
    int,
    float,
    char,
    string,
    bool,

    const Self = @This();

    pub fn fromToken(data: l.TokenData) ?Self {
        return switch (data) {
            .t_int => .int,
            .t_float => .float,
            .t_char => .char,
            .t_string => .string,
            .t_bool => .bool,
            else => null,
        };
    }
};

pub const ArrayType = struct {
    type: *const Type, // allow for nested arrays
    len: u32,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, typ: Type, len: u32) Self {
        const a_type = alloc.create(Type) catch |err| panic(err);
        a_type.* = typ;
        return Self{
            .type = a_type,
            .len = len,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        alloc.destroy(self.type);
    }
};

pub const Type = union(enum) {
    primitive: PrimitiveType,
    array: ArrayType,

    const Self = @This();

    pub fn initPrimitive(prim: PrimitiveType) Self {
        return Self{
            .primitive = prim,
        };
    }

    pub fn initArray(alloc: std.mem.Allocator, typ: Type, len: u32) Self {
        const arr = ArrayType.init(alloc, typ, len);
        return Self{
            .array = arr,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .array => |a| a.deinit(alloc),
            else => {},
        }
    }
};

pub const ArrayLiteral = struct {
    items: []const Expr,
    len: u16,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, items: []const Expr, len: u16) Self {
        const a_items = alloc.dupe(Expr, items) catch |err| panic(err);
        return Self{
            .items = a_items,
            .len = len,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        for (self.items) |itm| {
            itm.deinit(alloc);
        }

        alloc.free(self.items);
    }
};

pub const Primitive = union(PrimitiveType) {
    int: i32,
    float: f64,
    char: u8,
    string: []const u8, // owned slice
    bool: bool,

    const Self = @This();

    pub fn init(comptime kind: PrimitiveType, v: anytype) Self {
        return @unionInit(Primitive, @tagName(kind), v);
    }

    pub fn initString(alloc: std.mem.Allocator, slc: []const u8) Self {
        const s = alloc.dupe(u8, slc) catch |err| panic(err);
        return init(.string, s);
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .string => alloc.free(self.string),
            else => return,
        }
    }
};

pub const Typecast = struct {
    type: PrimitiveType,
    expr: Expr,

    const Self = @This();

    pub fn init(typ: PrimitiveType, exp: Expr) Self {
        return Self{ .type = typ, .expr = exp };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.expr.deinit(alloc);
    }
};

pub const BinaryExpr = struct {
    lhs: Expr,
    op: Operator,
    rhs: Expr,

    const Self = @This();

    pub fn init(lhs: Expr, op: Operator, rhs: Expr) Self {
        return Self{
            .lhs = lhs,
            .op = op,
            .rhs = rhs,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.lhs.deinit(alloc);
        self.rhs.deinit(alloc);
    }
};

pub const LiteralKind = enum {
    primitive,
    array,
};

pub const Literal = union(LiteralKind) {
    primitive: Primitive,
    array: ArrayLiteral,

    const Self = @This();

    pub fn init(comptime kind: LiteralKind, v: anytype) Self {
        return @unionInit(Literal, @tagName(kind), v);
    }

    pub fn initPrimitive(comptime kind: PrimitiveType, v: anytype) Self {
        return init(.primitive, Primitive.init(kind, v));
    }

    pub fn initPrimitiveString(alloc: std.mem.Allocator, slc: []const u8) Self {
        return init(.primitive, Primitive.initString(alloc, slc));
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .primitive => |p| p.deinit(alloc),
            .array => unreachable, // FIXME:
        }
    }
};

pub const ArrayIndex = struct {
    expr: Expr,
    idx: u32,

    const Self = @This();

    pub fn init(expr: Expr, idx: u32) Self {
        return Self{
            .expr = expr,
            .idx = idx,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.expr.deinit(alloc);
    }
};

pub const FunctionCall = struct {
    name: Expr, // just assume everything is callable
    args: []const Expr,

    const Self = @This();

    pub fn init(name: Expr, args: []const Expr, alloc: std.mem.Allocator) Self {
        const a_args = alloc.dupe(Expr, args) catch |err| panic(err);
        return Self{
            .name = name,
            .args = a_args,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        for (self.args) |arg| {
            arg.deinit(alloc);
        }
        alloc.free(self.args);
        self.name.deinit(alloc);
    }
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

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.ident.deinit(alloc);
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

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
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
    e_function_call: *const FunctionCall,
    e_binary: *const BinaryExpr,

    /// Creates an ExprData with the union's value heap-allocated for the AST.
    pub fn init(alloc: std.mem.Allocator, comptime kind: ExprKind, v: anytype) ExprData {
        const res = alloc.create(@TypeOf(v)) catch |err| panic(err);
        res.* = v;
        return @unionInit(ExprData, @tagName(kind), res);
    }

    pub fn deinit(self: *const ExprData, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .e_literal => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_negation => |v| alloc.destroy(v),
            .e_not => |v| alloc.destroy(v),
            .e_grouping => |v| alloc.destroy(v),
            .e_lvalue => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_typecast => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_array_literal => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_array_index => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_function_call => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
            .e_binary => |v| {
                v.deinit(alloc);
                alloc.destroy(v);
            },
        }
    }
};

pub const Expr = struct {
    span: *const SourceSpan,
    data: ExprData,

    pub fn init(alloc: std.mem.Allocator, comptime kind: ExprKind, v: anytype, loc: *const SourceSpan) Expr {
        const data = ExprData.init(alloc, kind, v);
        return Expr{ .span = loc, .data = data };
    }

    pub fn deinit(self: *const Expr, alloc: std.mem.Allocator) void {
        return self.data.deinit(alloc);
    }

    // useful utility methods
    pub fn initPrimitive(alloc: std.mem.Allocator, comptime kind: PrimitiveType, v: anytype, span: *const SourceSpan) Expr {
        const data = ExprData.init(alloc, .e_literal, Literal.initPrimitive(kind, v));
        return Expr{ .span = span, .data = data };
    }

    pub fn initIdent(alloc: std.mem.Allocator, v: []const u8, span: *const SourceSpan) Expr {
        const newv = alloc.dupe(u8, v) catch |err| panic(err);
        const data = ExprData.init(alloc, .e_lvalue, Lvalue{ .ident = newv });
        return Expr{ .span = span, .data = data };
    }

    pub fn initBinary(alloc: std.mem.Allocator, left: Expr, op: Operator, right: Expr, span: *const SourceSpan) Expr {
        const new = BinaryExpr.init(left, op, right);
        const data = ExprData.init(alloc, .e_binary, new);
        return Expr{
            .data = data,
            .span = span,
        };
    }

    pub fn initGrouping(alloc: std.mem.Allocator, inner: Expr, span: *const SourceSpan) Expr {
        const data = ExprData.init(alloc, .e_grouping, inner);
        return Expr{
            .data = data,
            .span = span,
        };
    }

    pub fn initNot(alloc: std.mem.Allocator, inner: Expr, span: *const SourceSpan) Expr {
        const data = ExprData.init(alloc, .e_not, inner);
        return Expr{
            .data = data,
            .span = span,
        };
    }

    pub fn initNegation(alloc: std.mem.Allocator, inner: Expr, span: *const SourceSpan) Expr {
        const data = ExprData.init(alloc, .e_negation, inner);
        return Expr{
            .data = data,
            .span = span,
        };
    }

    pub fn initTypecast(alloc: std.mem.Allocator, typ: PrimitiveType, expr: Expr, span: *const SourceSpan) Expr {
        const tc = Typecast.init(typ, expr);
        const data = ExprData.init(alloc, .e_typecast, tc);
        return Expr{
            .data = data,
            .span = span,
        };
    }
};

pub const PrintStmt = struct {
    items: []const Expr, // owned

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, items: []const Expr) Self {
        const new_items = alloc.dupe(Expr, items) catch |err| panic(err);
        return PrintStmt{ .items = new_items };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        for (self.items) |itm| {
            itm.deinit(alloc);
        }
        alloc.free(self.items);
    }
};

pub const ReadStmt = struct {
    ident: *const Lvalue,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, ident: Lvalue) Self {
        const lv = alloc.create(Lvalue) catch |err| panic(err);
        lv.* = ident;
        return Self{ .ident = lv };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.ident.deinit(alloc);
        alloc.destroy(self.ident);
    }
};

pub const ConstStmt = struct {
    ident: []const u8,
    value: Expr,
    exp: bool, // export

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, ident: []const u8, value: Expr, exp: bool) Self {
        const a_ident = alloc.dupe(u8, ident) catch |err| panic(err);
        return Self{
            .ident = a_ident,
            .value = value,
            .exp = exp,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.value.deinit(alloc);
        alloc.free(self.ident);
    }
};

pub const VarStmt = struct {
    ident: []const u8,
    typ: ?Type,
    value: ?Expr,
    exp: bool,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator, ident: []const u8, typ: ?Type, value: ?Expr, exp: bool) Self {
        const a_ident = alloc.dupe(u8, ident) catch |err| panic(err);
        return Self{
            .ident = a_ident,
            .typ = typ,
            .value = value,
            .exp = exp,
        };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        if (self.typ) |t| {
            t.deinit(alloc);
        }

        if (self.value) |e| {
            e.deinit(alloc);
        }

        alloc.free(self.ident);
    }
};

pub const AssignStmt = struct {
    lv: Lvalue,
    value: Expr,

    const Self = @This();

    pub fn init(lv: Lvalue, value: Expr) Self {
        return Self{ .lv = lv, .value = value };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.lv.deinit(alloc);
        self.value.deinit(alloc);
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
        alloc.free(self.stmts);
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

    pub fn init(alloc: std.mem.Allocator, comptime kind: StatementKind, v: anytype) StatementData {
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

    pub fn init(alloc: std.mem.Allocator, comptime kind: StatementKind, v: anytype, span: *const SourceSpan) Statement {
        const res = StatementData.init(alloc, kind, v);
        return Statement{ .data = res, .span = span };
    }

    pub fn deinit(self: *const Self, alloc: std.mem.Allocator) void {
        self.data.deinit(alloc);
    }

    pub fn initPrintStmt(alloc: std.mem.Allocator, items: []const Expr, span: *const SourceSpan) Statement {
        const res = PrintStmt.init(alloc, items);
        return Statement{ .data = .{ .s_print = res }, .span = span };
    }

    pub fn initReadStmt(alloc: std.mem.Allocator, lv: Lvalue, span: *const SourceSpan) Statement {
        const res = ReadStmt.init(alloc, lv);
        return Statement{ .data = .{ .s_read = res }, .span = span };
    }

    pub fn initConstStmt(alloc: std.mem.Allocator, id: []const u8, val: Expr, exp: bool, span: *const SourceSpan) Statement {
        const res = ConstStmt.init(alloc, id, val, exp);
        return Statement{ .data = .{ .s_const = res }, .span = span };
    }

    pub fn initVarStmt(alloc: std.mem.Allocator, id: []const u8, typ: ?Type, val: ?Expr, exp: bool, span: *const SourceSpan) Statement {
        const res = VarStmt.init(alloc, id, typ, val, exp);
        return Statement{
            .data = .{
                .s_var = res,
            },
            .span = span,
        };
    }

    pub fn initAssignStmt(lv: Lvalue, exp: Expr, span: *const SourceSpan) Statement {
        const res = AssignStmt.init(lv, exp);
        return Statement{
            .data = .{
                .s_assign = res,
            },
            .span = span,
        };
    }
};

test "make expr" {
    // TODO: update tests
    const alloc = std.heap.page_allocator;
    const expr = ExprData.init(.e_literal, alloc, Literal.initPrimitive(.{ .int = 234 }));
    const w = std.io.getStdErr().writer().any();
    var p = printer.AstPrinter.init(alloc, w);
    p.visitExpr(expr);
    try w.writeByte('\n');
    expr.deinit(alloc);
}
