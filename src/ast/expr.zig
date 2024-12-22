const std = @import("std");
const literal = @import("./literal.zig");
const ident = @import("./ident.zig");
const fncall = @import("fncall.zig");

const Literal = literal.Literal;
const Identifier = ident.Identifier;
const FnCall = fncall.FnCall;

pub const ArithmeticOperator = enum {
    op_add,
    op_sub,
    op_mul,
    op_div,
};

pub const ComparisonOperator = enum {
    op_gt,
    op_lt,
    op_eq,
    op_geq,
    op_leq,
    op_neq,
};

pub const LogicalOperator = enum {
    op_not,
    op_and,
    op_or,
};

pub const OperatorKind = enum {
    arithmetic,
    comparison,
    logical,
};

pub const Operator = union(OperatorKind) {
    arithmetic: ArithmeticOperator,
    comparison: ComparisonOperator,
    logical: LogicalOperator,
};

// TODO: lhs and rhs should be Expr
pub const BinaryExpr = struct {
    lhs: Literal,
    rhs: Literal,
    op: Operator,
};

// TODO: lhs and rhs should be Expr
pub const UnaryExpr = struct {
    rhs: Literal,
    op: Operator,
};

pub const ExprKind = enum {
    binary,
    unary,
    literal,
    ident,
    fncall,
};

pub const Expr = union(ExprKind) {
    binary: BinaryExpr,
    unary: UnaryExpr,
    literal: Literal,
    ident: Identifier,
    fncall: FnCall,
};

pub const Grouping = struct {
    inner: Expr,
};
