// unlike the cursed ultra hyper absolute cinema pro max™ python impl,
// these types will only represent ast types.

pub const Operator = enum { assign, equal, less_than, greater_than, less_than_or_equal, greater_than_or_equal, not_equal, mul, div, add, sub };

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

pub const BCType = enum {
    v_int,
    v_float,
    v_char,
    v_string,
    v_bool,
    v_array,
};

pub const BCArray = struct {
    items: []BCValue,
    len: Expr,
};

pub const BCValue = union(BCType) {
    v_int: i32,
    v_float: f64,
    v_char: u8,
    v_string: []const u8, // owned slice
    v_bool: bool,
    v_array: BCArray,
};

pub const ExprKind = enum {
    e_literal,
    e_negration,
    e_not,
    e_grouping,
    e_ident,
    e_typecast,
    e_array_literal,
    e_binary,
    e_array_index,
};

pub const Typecast = struct {
    typ: BCType,
    expr: Expr,
};

pub const BinaryExpr = struct {
    lhs: Expr,
    op: Operator,
    rhs: Expr,
};

pub const ArrayIndex = struct {
    ident: []const u8, // owned slice
    idx: i32,
};

pub const Expr = union(ExprKind) {
    e_literal: Expr,
    e_not: Expr,
    e_grouping: Expr,
    e_ident: []const u8, // owned slice
    e_typecast: Typecast,
    e_array_literal: []Expr,
    e_binary: BinaryExpr,
    e_array_index: ArrayIndex,
};

pub const StatementKind = enum {
    s_print,
    s_read,
    s_const,
    s_var,
    s_assign,

    s_program,
};

pub const PrintStmt = struct {
    items: []Expr, // owned
};

pub const ReadStmt = struct {
    ident: []const u8,
};

pub const ConstStmt = struct {
    ident: []const u8,
    value: Expr,
    exp: bool,
};

pub const VarStmt = struct {
    ident: []const u8,
    typ: BCType,
    exp: bool,
    expr: ?Expr,
};

pub const AssignStmt = struct {
    ident: []const u8,
    value: Expr,
};

pub const ArrayIndexAssignStmt = struct {
    lhs: ArrayIndex,
    value: Expr,
};

pub const Program = struct {
    stmts: []Statement, // owned
};

pub const Statement = struct {
    s_print: PrintStmt,
    s_read: ReadStmt,
    s_const: ConstStmt,
    s_var: VarStmt,
    s_assign: AssignStmt,
    s_array_index_assign: ArrayIndexAssignStmt,
    s_program: Program,
};
