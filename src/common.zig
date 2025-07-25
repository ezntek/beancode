pub const CompilerStage = enum {
    lexer,
    parser,
    // TODO: add more
};

pub const Location = struct {
    file_name: []const u8,
    bol: u32,
    line: u32,
    col: u32,
};
