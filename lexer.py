import typing as t
from dataclasses import dataclass

import util

Keyword = t.Literal[
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
    "div",
    "mod",
]

# lexer types
LType = t.Literal["number", "boolean", "string", "char", "array"]

Type = t.Literal["integer", "real", "boolean", "string", "char", "array"]

Operator = t.Literal[
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
]

Separator = t.Literal[
    "left_paren",
    "right_paren",
    "left_bracket",
    "right_bracket",
    "left_curly",
    "right_curly",
    "colon",
    "comma",
    "dot",
]

_SEPARATORS = {
    "left_curly": "{",
    "right_curly": "}",
    "left_bracket": "[",
    "right_bracket": "]",
    "left_paren": "(",
    "right_paren": ")",
    "colon": ":",
    "comma": ",",
    "dot": "."
}

_OPERATORS = {
    "assign": "<-",
    "less_than_or_equal": "<=",
    "greater_than_or_equal": ">=",
    "not_equal": "<>",
    "greater_than": ">",
    "less_than": "<",
    "equal": "=",
    "mul": "*",
    "add": "+",
    "sub": "-",
    "div": "/"
}

@dataclass
class Literal:
    kind: LType
    value: str


TokenType = t.Literal[
    "newline", "keyword", "ident", "literal", "operator", "separator", "type"
]


@dataclass
class Token:
    kind: TokenType
    pos: tuple[int, int, int]
    keyword: Keyword | None = None
    ident: str | None = None
    literal: Literal | None = None
    operator: Operator | None = None
    separator: Separator | None = None
    typ: Type | None = None

    # content, kind
    def get(self) -> tuple[str, TokenType]:
        if self.keyword != None:
            return (self.keyword, "keyword")
        elif self.ident != None:
            return (self.ident, "ident")
        elif self.literal != None:
            return (self.literal.value, "literal")
        elif self.operator != None:
            return (self.operator, "operator")
        elif self.separator != None:
            return (self.separator, "separator")
        elif self.typ != None:
            return (self.typ, "type")
        elif self.kind == "newline":
            return ("\n", "newline")
        else:
            raise Exception() # TODO: fix

    # content, kind
    def get_raw(self) -> tuple[str, TokenType]:
        if self.keyword != None:
            return (self.keyword, "keyword")
        elif self.ident != None:
            return (self.ident, "ident")
        elif self.literal != None:
            return (self.literal.value, "literal")
        elif self.operator != None:
            return (_OPERATORS[self.operator], "operator")
        elif self.separator != None:
            return (_SEPARATORS[self.separator], "separator")
        elif self.typ != None:
            return (self.typ, "type")
        elif self.kind == "newline":
            return ("\n", "newline")
        else:
            raise Exception() # TODO: fix

    def __repr__(self) -> str:
        s, kind = self.get()
        s = s if s != "\n" else ""
        return f"token({kind}): {s}"

class Lexer:
    file: str
    cur: int
    bol: int
    row: int
    keywords: list[str]
    types: list[str]
    res: list[Token]

    def __init__(self, file: str) -> None:
        self.cur = 0
        self.bol = 0
        self.row = 1
        self.file = file
        self.res = []
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
        ]
        self.types = ["integer", "real", "boolean", "string", "char", "array"]

    def get_pos(self, back=1) -> tuple[int, int, int]:
        row = self.row
        col = self.cur - self.bol
        return (row, col + 1 - back, self.bol)

    def is_separator(self, ch: str) -> bool:
        return ch in "{}[]():,."

    def is_operator_start(self, ch: str) -> bool:
        return ch in "+-*/=<>"

    def is_numeral(self, potential_num: str) -> bool:
        if len(potential_num) == 1 and potential_num[0] == "-":
            return False

        if potential_num[0] == "-" and not potential_num[1].isdigit():
            return False
        elif potential_num[0] == "-" and potential_num[1].isdigit():
            potential_num = potential_num[1:]

        for ch in potential_num:
            if not ch.isdigit() and ch != "_" and ch != ".":
                return False
        return True

    def is_keyword(self, s: str) -> bool:
        return s.lower() in self.keywords

    def is_type(self, s: str) -> bool:
        return s.lower() in self.types

    def trim_left(self):
        if self.cur >= len(self.file):
            return

        while self.cur < len(self.file) and (
            self.file[self.cur].isspace() and self.file[self.cur] != "\n"
        ):
            self.cur += 1

        return self.trim_comment()

    def trim_comment(self):
        if self.cur + 2 > len(self.file):
            return

        if self.file[self.cur : self.cur + 2] == "//":
            self.cur += 2
            while self.cur < len(self.file) and self.file[self.cur] != "\n":
                self.cur += 1
            self.cur += 1  # get rid of newline

            return self.trim_left()

        if self.file[self.cur : self.cur + 2] == "/*":
            self.cur += 2

            while (
                self.cur < len(self.file) and self.file[self.cur : self.cur + 2] != "*/"
            ):
                if self.file[self.cur] == "\n":
                    self.row += 1
                    self.bol = self.cur + 1
                self.cur += 1

            # when we find */, we must skip 2 past to avoid pasing it
            self.cur += 2

            return self.trim_left()

    def next_operator(self) -> Token | None:
        if self.cur + 2 < len(self.file):
            cur_pair = self.file[self.cur : self.cur + 2]

            hm = {
                "<-": "assign",
                "<=": "less_than_or_equal",
                ">=": "greater_than_or_equal",
                "<>": "not_equal",
            }

            res = hm.get(cur_pair)

            if res is not None:
                self.cur += 2
                return Token("operator", self.get_pos(back=2), operator=res)  # type: ignore

        hm = {
            ">": "greater_than",
            "<": "less_than",
            "=": "equal",
            "*": "mul",
            "+": "add",
            "-": "sub",
            "/": "div",
        }

        res = hm.get(self.file[self.cur])

        if res is not None:
            if res == "sub" and self.file[self.cur + 1].isdigit():
                return None

            self.cur += 1
            return Token("operator", self.get_pos(), operator=res)  # type: ignore

    def next_separator(self) -> Token | None:
        hm = {
            "{": "left_curly",
            "}": "right_curly",
            "[": "left_bracket",
            "]": "right_bracket",
            "(": "left_paren",
            ")": "right_paren",
            ":": "colon",
            ",": "comma",
            ".": "dot",
        }

        res = hm.get(self.file[self.cur])

        if res is not None:
            self.cur += 1
            return Token("separator", self.get_pos(), separator=res)  # type: ignore

    def next_word(self) -> str:
        # do not question why this works. its sorcery from the old zig shit

        begin = self.cur
        curr_ch = self.file[self.cur]
        end = self.cur
        string_or_char_literal = curr_ch == "'" or curr_ch == '"'
        begin_ch = curr_ch

        if not string_or_char_literal:
            if curr_ch == "-":
                self.cur += 1  # skip past
                end += 1
                curr_ch = self.file[self.cur]

                if (
                    len(self.res) != 0
                    and self.res[len(self.res) - 1].kind in ["ident", "literal"]
                    or self.res[len(self.res) - 1].separator
                    in ["right_bracket", "right_paren", "right_curly"]
                ):
                    res = self.file[begin:end]
                    return res

            while (
                not curr_ch.isspace()
                and not self.is_separator(curr_ch)
                and not self.is_operator_start(curr_ch)
            ):
                self.cur += 1
                end += 1

                if self.cur >= len(self.file):
                    break

                curr_ch = self.file[self.cur]
        else:
            # skip past the initial delim
            self.cur += 1
            end += 1
            curr_ch = self.file[self.cur]

            while curr_ch != begin_ch:
                self.cur += 1
                end += 1

                if self.cur >= len(self.file):
                    break

                curr_ch = self.file[self.cur]

            # account for ending quote
            self.cur += 1
            end += 1

        res = self.file[begin:end]
        return res

    def next_keyword(self, word: str) -> Token | None:
        if self.is_keyword(word.lower()):
            return Token("keyword", (self.row, self.cur - self.bol - len(word), self.bol), keyword=word.lower())  # type: ignore
        else:
            return None

    def next_type(self, typ: str) -> Token | None:
        if typ.lower() in ["integer", "string", "boolean", "real", "array", "char"]:
            return Token("type", (self.row, self.cur - self.bol - len(typ), self.bol), typ=typ.lower())  # type: ignore
        else:
            return None

    def next_string_or_char_literal(self, word: str) -> Token | None:
        if word[0] == '"' and word[len(word) - 1] == '"':
            slc = word[1 : len(word) - 1]
            return Token(
                "literal",
                (self.row, self.cur - self.bol - (len(word) + 2), self.bol),
                literal=Literal("string", slc),
            )

        if word[0] == "'" and word[len(word) - 1] == "'":
            if len(word) > 3:
                row = self.row
                col = self.cur - self.bol - len(word)
                util.BCError(
                    f"char literal cannot contain more than 1 char", (row, col, self.bol)
                )

            # -3 for begin delim, char and end delim
            return Token(
                "literal",
                (self.row, self.cur - self.bol - 3, self.bol),
                literal=Literal("char", word[1]),
            )

        return None

    def next_boolean(self, word: str) -> str | None:
        return word if word.lower() in ["true", "false"] else None

    def next_token(self) -> Token | None:
        self.trim_left()

        if self.cur >= len(self.file):
            return None

        t: Token | None

        cur = self.file[self.cur]
        if cur == "\n":
            self.row += 1
            self.bol = self.cur + 1
            self.cur += 1
            return Token("newline", self.get_pos())

        if (t := self.next_operator()) is not None:
            return t

        if (t := self.next_separator()) is not None:
            return t

        word = self.next_word()

        if len(self.res) != 0:
            should_be_sub = lambda *_: self.res[len(self.res) - 1].kind in [
                "ident",
                "literal",
            ] or self.res[len(self.res) - 1].separator in [
                "right_bracket",
                "right_paren",
                "right_curly",
            ]

            if len(word) == 1 and should_be_sub() and word[0] == "-":
                return Token("operator", self.get_pos(), operator="sub")

        if self.is_numeral(word):
            return Token("literal", (self.row, self.cur - self.bol - len(word) + 1, self.bol), literal=Literal("number", word))
        elif word[0] == "-" and not word[1].isdigit():  # scuffed
            self.cur += 1
            return Token("operator", self.get_pos(), operator="sub")

        if t := self.next_keyword(word):
            return t
        if t := self.next_type(word):
            return t
        if t := self.next_string_or_char_literal(word):
            return t

        b: str | None
        if (b := self.next_boolean(word)) is not None:
            return Token("literal", (self.row, self.cur - self.bol - len(word) + 1, self.bol), literal=Literal("boolean", b))  # type: ignore

        return Token("ident", (self.row, self.cur - self.bol - len(word) + 1, self.bol), ident=word)

    def tokenize(self) -> list[Token]:
        self.res = []

        while self.cur < len(self.file):
            t: Token | None
            if (t := self.next_token()) is not None:
                self.res.append(t)
            else:
                break

        self.res.append(Token("newline", self.get_pos()))
        return self.res
