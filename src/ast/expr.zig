const std = @import("std");
const literal = @import("./literal.zig");
const ident = @import("./ident.zig");
const fncall = @import("fncall.zig");

const Literal = literal.Literal;
const Identifier = ident.Identifier;
const FnCall = fncall.FnCall;

const ArithmeticOperator = enum {
    op_add,
    op_sub,
    op_mul,
    op_div,
};

const ComparisonOperator = enum {
    op_gt,
    op_lt,
    op_eq,
    op_geq,
    op_leq,
    op_neq,
};

const LogicalOperator = enum {
    op_not,
    op_and,
    op_or,
};

const OperatorKind = enum {
    arithmetic,
    comparison,
    logical,
};

const Operator = union(OperatorKind) {
    arithmetic: ArithmeticOperator,
    comparison: ComparisonOperator,
    logical: LogicalOperator,
};

const BinaryExpr = struct {
    lhs: Expr,
    rhs: Expr,
    op: Operator,
};

const UnaryExpr = struct {
    rhs: Expr,
    op: Operator,
};

const ExprKind = enum {
    binary,
    unary,
    literal,
    ident,
    fncall,
};

const Expr = union(ExprKind) {
    binary: BinaryExpr,
    unary: UnaryExpr,
    literal: Literal,
    ident: Identifier,
    fncall: FnCall,
};

const Grouping = struct {
    inner: Expr,
};
