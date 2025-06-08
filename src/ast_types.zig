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

pub const BCPrimitiveType = enum {
    int,
    float,
    char,
    string,
    bool,
};

pub const BCArrayType = struct {
    type: *const BCType, // allow for nested arrays
    len: u32,
};

pub const BCType = union(enum) {
    primitive: BCPrimitiveType,
    v_array: BCArrayType,
};

// array literal
pub const BCArrayLiteral = struct {
    items: []BCValue,
    len: Expr,
};

pub const BCValue = union(BCPrimitiveType) {
    int: i32,
    float: f64,
    char: u8,
    string: []const u8, // owned slice
    bool: bool,
};

pub const Typecast = struct {
    typ: BCPrimitiveType,
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

pub const Expr = union(enum) {
    e_literal: *const BCValue,
    e_negation: *const Expr,
    e_not: *const Expr,
    e_grouping: *const Expr,
    e_ident: []const u8,
    e_typecast: *const Typecast,
    e_array_literal: *const BCArrayLiteral,
    e_binary: *const BinaryExpr,
    e_array_index: *const ArrayIndex,
};

pub const PrintStmt = struct {
    items: []const Expr, // owned
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

pub const Statement = union(enum) {
    s_print: PrintStmt,
    s_read: ReadStmt,
    s_const: ConstStmt,
    s_var: VarStmt,
    s_assign: AssignStmt,
    s_array_index_assign: ArrayIndexAssignStmt,
    s_program: Program,
};
