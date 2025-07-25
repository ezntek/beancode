// unlike the cursed ultra hyper absolute cinema pro max™ python impl,
// these types will only represent ast types.

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

pub const Expr = union(enum) {
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
