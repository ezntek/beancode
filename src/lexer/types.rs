use std::fmt;

pub enum Keyword {
    Declare,
    Constant,
    Output,
    Input,
    And,
    Or,
    Not,
    If,
    Then,
    Elseif,
    Else,
    Endif,
    Case,
    Of,
    Otherwise,
    Endcase,
    While,
    Do,
    Endwhile,
    Repeat,
    Until,
    For,
    To,
    Step,
    Next,
    Procedure,
    Endprocedure,
    Call,
    Function,
    Return,
    Returns,
    Endfunction,
    Openfile,
    Readfile,
    Writefile,
    Closefile,
    Include,
}

pub enum Separator {
    Lparen,
    Rparen,
    Lbracket,
    Rbracket,
    Lcurly,
    Rcurly,
    Colon,
    Comma,
    Dot,
}

pub enum Operator {
    Assign,
    Leq,
    Geq,
    Ne,
    Gt,
    Lt,
    Eq,
    Mul,
    Add,
    Sub,
    Div,
}

pub enum Type {
    Integer,
    Real,
    Char,
    Boolean,
    String,
    Array,
    // extra functionality
    Float,
}

pub enum TokenKind {
    Keyword(Keyword),
    Type(Type),
    Ident(String),
    Literal(String),
    Separator(Separator),
    Operator(Operator),
    Newline,
    Eof,
}

pub struct Token {
    pub kind: TokenKind,
    pub pos: (usize, usize, usize), // line, col, bol
}

impl Token {
    pub fn new(kind: TokenKind, pos: (usize, usize, usize)) -> Self {
        Self { kind, pos }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TokenKind::*;
        let (content, kind) = match &self.kind {
            Keyword(kw) => (Some(kw.to_string()), "keyword"),
            Type(t) => (Some(t.to_string()), "type"),
            Ident(i) => (Some(i.clone()), "ident"),
            Literal(l) => (Some(l.clone()), "literal"),
            Separator(sep) => (Some(sep.to_string()), "separator"),
            Operator(op) => (Some(op.to_string()), "operator"),
            Newline => (None, "newline"),
            Eof => (None, "eof"),
        };

        if let Some(c) = content {
            write!(f, "token({kind}): `{c}`")
        } else {
            write!(f, "token({kind})")
        }
    }
}

impl From<String> for Keyword {
    fn from(value: String) -> Self {
        use Keyword::*;
        match value.as_str() {
            "declare" => Declare,
            "constant" => Constant,
            "output" => Output,
            "input" => Input,
            "and" => And,
            "or" => Or,
            "not" => Not,
            "if" => If,
            "then" => Then,
            "else" => Else,
            "endif" => Endif,
            "case" => Case,
            "of" => Of,
            "endcase" => Endcase,
            "otherwise" => Otherwise,
            "while" => While,
            "do" => Do,
            "endwhile" => Endwhile,
            "repeat" => Repeat,
            "until" => Until,
            "for" => For,
            "to" => To,
            "step" => Step,
            "next" => Next,
            "procedure" => Procedure,
            "endprocedure" => Endprocedure,
            "call" => Call,
            "function" => Function,
            "return" => Return,
            "returns" => Returns,
            "endfunction" => Endfunction,
            "openfile" => Openfile,
            "readfile" => Readfile,
            "writefile" => Writefile,
            "closefile" => Closefile,
            "include" => Include,
            "elseif" => Elseif,
            _ => panic!("invalid string `{value}` passed as input for token!"),
        }
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Keyword::*;
        let s = match self {
            Declare => "declare",
            Constant => "constant",
            Output => "output",
            Input => "input",
            And => "and",
            Or => "or",
            Not => "not",
            If => "if",
            Then => "then",
            Else => "else",
            Endif => "endif",
            Case => "case",
            Of => "of",
            Endcase => "endcase",
            Otherwise => "otherwise",
            While => "while",
            Do => "do",
            Endwhile => "endwhile",
            Repeat => "repeat",
            Until => "until",
            For => "for",
            To => "to",
            Step => "step",
            Next => "next",
            Procedure => "procedure",
            Endprocedure => "endprocedure",
            Call => "call",
            Function => "function",
            Return => "return",
            Returns => "returns",
            Endfunction => "endfunction",
            Openfile => "openfile",
            Readfile => "readfile",
            Writefile => "writefile",
            Closefile => "closefile",
            // extra functionality
            Include => "include",
            Elseif => "elseif",
        };

        write!(f, "{s}")
    }
}

impl From<char> for Separator {
    fn from(value: char) -> Self {
        use Separator::*;
        match value {
            '(' => Lparen,
            ')' => Rparen,
            '[' => Lbracket,
            ']' => Rparen,
            '{' => Lcurly,
            '}' => Rcurly,
            ',' => Comma,
            ':' => Colon,
            '.' => Dot,
            _ => panic!("invalid char '{value}' passed in for separator"),
        }
    }
}

impl fmt::Display for Separator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Separator::*;
        let ch = match self {
            Lparen => '(',
            Rparen => ')',
            Lbracket => '[',
            Rbracket => ']',
            Lcurly => '{',
            Rcurly => '}',
            Comma => ',',
            Colon => ':',
            Dot => '.',
        };
        write!(f, "{}", ch)
    }
}

impl From<String> for Operator {
    fn from(value: String) -> Self {
        use Operator::*;
        match value.as_str() {
            "<-" => Assign,
            "<=" => Leq,
            ">=" => Geq,
            "<>" => Ne,
            ">" => Gt,
            "<" => Lt,
            "=" => Eq,
            "*" => Mul,
            "+" => Add,
            "-" => Sub,
            "/" => Div,
            _ => panic!("invalid operator string: '{}'", value),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Operator::*;
        let symbol = match self {
            Assign => "<-",
            Leq => "<=",
            Geq => ">=",
            Ne => "<>",
            Gt => ">",
            Lt => "<",
            Eq => "=",
            Mul => "*",
            Add => "+",
            Sub => "-",
            Div => "/",
        };
        write!(f, "{}", symbol)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Type::*;
        let name = match self {
            Integer => "integer",
            Real => "real",
            Char => "char",
            Boolean => "boolean",
            String => "string",
            Array => "array",
            Float => "float",
        };
        write!(f, "{}", name)
    }
}

impl From<String> for Type {
    fn from(value: String) -> Self {
        use Type::*;
        match value.to_lowercase().as_str() {
            "integer" => Integer,
            "real" => Real,
            "char" => Char,
            "boolean" => Boolean,
            "string" => String,
            "array" => Array,
            "float" => Float,
            _ => panic!("invalid type string: '{}'", value),
        }
    }
}
