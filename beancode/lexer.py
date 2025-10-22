import typing

from dataclasses import dataclass

from beancode.bean_ast import BCPrimitiveType, Identifier
from .error import *
from . import Pos, __version__, is_case_consistent, panic

TokenKind = typing.Literal[
    "declare",
    "constant",
    "output",
    "input",
    "and",
    "or",
    "not",
    "if",
    "then",
    "else",
    "endif",
    "case",
    "of",
    "otherwise",
    "endcase",
    "while",
    "do",
    "endwhile",
    "repeat",
    "until",
    "for",
    "to",
    "step",
    "next",
    "procedure",
    "endprocedure",
    "call",
    "function",
    "return",
    "returns",
    "endfunction",
    "openfile",
    "readfile",
    "writefile",
    "closefile",
    "newline",
    # extra feachur™
    "include",
    "include_ffi",
    "null",
    "export",
    "scope",
    "endscope",
    "print",
    "trace",
    # symbols
    "assign",
    "equal",
    "less_than",
    "greater_than",
    "less_than_or_equal",
    "greater_than_or_equal",
    "not_equal",
    "mul",
    "div",
    "add",
    "sub",
    "pow",
    "left_paren",
    "right_paren",
    "left_bracket",
    "right_bracket",
    "left_curly",
    "right_curly",
    "colon",
    "comma",
    "dot",
    "literal_string",
    "literal_char",
    "literal_number",
    "true",
    "false",
    "ident",
    "eof",
]

@dataclass
class Token:
    kind: TokenKind
    pos: Pos 
    data: str | None = None

    def print(self, file=sys.stdout):
        match self.kind:
            case "literal_string":
                s = f'"{self.data}"'
            case "true" | "false":
                s = f"<{self.data}>"
            case "literal_char":
                s = f"'{self.data}'"
            case "literal_number" | "ident":
                s = self.data
            case _:
                s = self.kind

        print(f"token[{self.pos}]: {s}", file=file)

    def __repr__(self) -> str:
        return f"token({self.kind})"


class Lexer:
    src: str
    row: int
    bol: int
    cur: int
    keywords: list[str]
    types: list[str]

    def __init__(self, src: str) -> None:
        self.cur = 0
        self.bol = 0
        self.row = 1
        self.src = src
        self.keywords = [
            "declare",
            "constant",
            "output",
            "input",
            "and",
            "or",
            "not",
            "if",
            "then",
            "else",
            "endif",
            "case",
            "of",
            "otherwise",
            "endcase",
            "while",
            "do",
            "endwhile",
            "repeat",
            "until",
            "for",
            "to",
            "step",
            "next",
            "procedure",
            "endprocedure",
            "call",
            "function",
            "returns",
            "return",
            "endfunction",
            "openfile",
            "readfile",
            "writefile",
            "closefile",
            # extra feature
            "trace",
            "scope",
            "endscope",
            "include",
            "include_ffi",
            "export",
            "print",
        ]
        self.types = ["integer", "real", "boolean", "string", "char", "array"]

    def reset(self):
        self.row = 0
        self.bol = 0
        self.cur = 0

    def get_cur(self):
        return self.src[self.cur]

    def peek(self):
        return self.src[self.cur + 1]

    def bump_newline(self):
        self.row += 1
        self.cur += 1
        self.bol = self.cur
   
    def in_bounds(self) -> bool:
        return self.cur < len(self.src)

    def pos(self, span: int) -> Pos:
        return Pos(row=self.row, col=self.cur - self.bol + 1 - span, span=span)

    def pos_here(self, span: int) -> Pos:
        return Pos(row=self.row, col=self.cur - self.bol + 1, span=span)

    def is_separator(self, ch: str) -> bool:
        return ch in "{}[]();:,"

    def is_operator_start(self, ch: str) -> bool:
        return ch in "+-*/<>=^"

    def trim_spaces(self) -> None:
        if not self.in_bounds():
            return

        cur = str()
        while self.in_bounds() and (cur := self.get_cur()).isspace() and cur != '\n':
            self.cur += 1

        self.trim_comments()

    def trim_comments(self) -> None:
        # if there are not 2 chars more in the stream (// and /*)
        if self.cur + 2 >= len(self.src):
            return

        pair = self.src[self.cur:self.cur+2]
        if pair not in ("//", "/*"):
            return

        if pair == "//":
            self.cur += 2 # skip past comment marker

            while self.in_bounds() and self.get_cur() != '\n':
                self.cur += 1

            self.trim_spaces()
        elif pair == "/*":
            self.cur += 2

            while self.in_bounds() and self.src[self.cur:self.cur+2] != '*/':
                if self.get_cur() == '\n':
                    self.bump_newline()
                else:
                    self.cur += 1

            # found */
            self.cur += 2
    
        self.trim_spaces()

    def next_double_symbol(self) -> Token | None:
        if not self.is_operator_start(self.get_cur()):
            return None

        if self.cur + 2 >= len(self.src):
            return None
 
        TABLE: dict[str, TokenKind] = {
            "<>": "not_equal",
            ">=": "greater_than_or_equal",
            "<=": "less_than_or_equal",
            "<-": "assign",
        }

        pair = self.src[self.cur:self.cur+2]
        kind = TABLE.get(pair)
        
        if kind is not None:
            self.cur += 2
            return Token(kind, self.pos(2))
    
    def next_single_symbol(self) -> Token | None:
        cur = self.get_cur()
        if not self.is_separator(cur) and not self.is_operator_start(cur):
            return None

        TABLE: dict[str, TokenKind] = {
            '{': "left_curly",
            '}': "right_curly",
            '[': "left_bracket",
            ']': "right_bracket",
            '(': "left_paren",
            ')': "right_paren",
            ':': "colon",
            ';': "newline",
            ',': "comma",
            '=': "equal",
            '<': "less_than",
            '>': "greater_than",
            '*': "mul",
            '/': "div",
            '+': "add",
            '-': "sub",
            '^': "pow",
        } 

        kind = TABLE.get(self.get_cur())
        if kind is not None:
            self.cur += 1
            return Token(kind, self.pos(1))

    def next_word(self) -> str:
        begin = self.cur
        len = 0
        DELIMS = "\"'"
        is_delimited_literal = self.src[begin] in DELIMS
        delim = str()
        if is_delimited_literal:
            delim = self.get_cur()
            self.cur += 1
            len += 1

        while True:
            stop = False
            cur = self.get_cur()
            if is_delimited_literal:
                stop = cur in ("\n\r"+DELIMS)
            else:
                stop = self.is_operator_start(cur) or self.is_separator(cur) or cur.isspace()

            if cur == '\\':
                len += 1
                self.cur += 1

            if stop or not self.in_bounds():
                break

            len += 1
            self.cur += 1

        if is_delimited_literal:
            len += 1
            cur = self.get_cur()
            if cur != delim:
                if cur in DELIMS:
                    raise BCError("mismatched delimiter in literal\n" + 
                        "did you insert the wrong ending quotation mark>", self.pos(len))
                else:
                    raise BCError("unterminated delimited literal\n"+
                        "did you forget to insert an ending quotation mark?", self.pos(len))
            else:
                self.cur += 1
        elif not self.in_bounds():
            # we don't set eof to true, becuase we do not allow for multile string literals, and this
            # would break the REPL.
            raise BCError("unexpected end of file while scanning for delimited literal", self.pos(len))

        res = self.src[begin:begin+len]
        return res

    def next_keyword(self, word: str) -> Token | None:
        if word not in self.keywords:
            return None
        
        kind: TokenKind = word # type: ignore
        return Token(kind, self.pos(len(word)))

    def _is_number(self, word: str) -> bool:
        found_decimal = False

        for ch in word:
            if ch.isdigit():
                continue

            if ch == '.':
                if found_decimal:
                    return False
                else:
                    found_decimal = True
                continue

            return False

        return True

    def next_literal(self, word: str) -> Token | None:
        if word[0] in "\"'":
            if len(word) == 1:
                panic("reached unreachable code")

            res = word[1:-1]
            kind: TokenKind = "literal_string" if word[0] == '"' else "literal_char"

            return Token(kind, self.pos(len(word)), data=res)

        if self._is_number(word):
            return Token("literal_number", self.pos(len(word)), data=word)

        if is_case_consistent(word):
            if word == "true":
                return Token("true", self.pos(len(word)))
            elif word == "false":
                return Token("false", self.pos(len(word)))

    def _is_ident(self, word: str) -> bool:
        if not word[0].isalpha():
            return False
    
        for ch in word:
            if not ch.isalnum() and ch not in "_.":
                return False

        return True

    def next_ident(self, word: str) -> Token:
        p = self.pos(len(word))
        if self._is_ident(word):
            return Token("ident", p, data=word)
        else:
            raise BCError("invalid identifier", p)

    def next_token(self) -> Token:
        self.trim_spaces()
        if not self.in_bounds():
            return Token("eof", self.pos(1))

        if self.get_cur() == '\n':
            return Token("newline", self.pos(1))

        res: Token | None
        if res := self.next_single_symbol():
            return res

        if res := self.next_double_symbol():
            return res
        
        word = self.next_word()

        if res := self.next_keyword(word):
            return res
            
        if res := self.next_literal(word):
            return res

        return self.next_ident(word)

    def tokenize(self) -> list[Token]:
        res = list()
        while self.in_bounds():
            res.append(self.next_token())
        return res
