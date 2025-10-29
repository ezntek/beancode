import sys
import os

from . import Pos


class BCError(Exception):
    pos: Pos | None
    eof: bool
    proc: str | None
    func: str | None

    def __init__(self, msg: str, pos: Pos | None = None, eof=False, proc=None, func=None) -> None:  # type: ignore
        self.eof = eof
        self.proc = proc
        self.func = func
        self.pos = pos

        s = f"\033[31;1merror: \033[0m{msg}\n"
        self.msg = s
        super().__init__(s)

    def print(self, filename: str, file_content: str):
        if self.pos is None:
            print(self.msg, end="", file=sys.stderr)
            sys.stderr.flush()
            return

        line_no = self.pos.row
        col = self.pos.col
        bol = 0

        i = 1
        j = -1
        while i < line_no and j < len(file_content):
            j += 1
            while file_content[j] != "\n":
                j += 1
            i += 1
        bol = j + 1

        eol = bol
        while eol != len(file_content) and file_content[eol] != "\n":
            eol += 1

        line_begin = f" \033[31;1m{line_no}\033[0m | "
        padding = len(str(line_no) + "  | ") + col - 1
        tabs = 0
        spaces = lambda *_: " " * padding + "\t" * tabs

        info = f"{filename}:{line_no}: "
        print(f"\033[0m\033[1m{info}", end="", file=sys.stderr)
        msg_lines = self.msg.splitlines()
        print(
            msg_lines[0], end="", file=sys.stderr
        )  # splitlines on a non-empty string guarantees one elem
        if len(msg_lines) == 1:
            print(file=sys.stderr)

        for msg_line in msg_lines[1:]:
            sp = " " * len(info)
            print(f"\033[2m\n{sp}{msg_line}\033[0m", file=sys.stderr)

        print(line_begin, end="", file=sys.stderr)
        print(file_content[bol:eol], file=sys.stderr)

        for ch in file_content[bol:eol]:
            if ch == "\t":
                padding -= 1
                tabs += 1

        tildes = f"{spaces()}\033[31;1m{'~' * self.pos.span}\033[0m"
        print(tildes, file=sys.stderr)

        indicator = f"{spaces()}\033[31;1m"
        if os.name == "nt":
            indicator += "+-"
        else:
            indicator += "∟"

        indicator += f" \033[0m\033[1merror at line {line_no} column {col}\033[0m"
        print(indicator, file=sys.stderr)
        sys.stderr.flush()


def info(msg: str):
    print(
        f"\033[34;1minfo:\033[0m {msg}",
        file=sys.stderr,
    )


def warn(msg: str):
    print(
        f"\033[33;1mwarn:\033[0m {msg}",
        file=sys.stderr,
    )


def error(msg: str):
    print(
        f"\033[31;1merror:\033[0m {msg}",
        file=sys.stderr,
    )
