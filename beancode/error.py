import sys
import os

from . import Pos

_bcerror_debug = False


class BCError(Exception):
    pos: Pos | None
    eof: bool
    proc: str | None
    func: str | None
    msg: str

    def __init__(
        self, msg: str, pos: Pos | None = None, eof=False, proc=None, func=None
    ) -> None:  # type: ignore
        self.eof = eof
        self.proc = proc
        self.func = func
        self.pos = pos

        s = f"\033[31;1merror: \033[0m{msg}\n"
        self.msg = s
        super().__init__(s)

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "msg": self.msg,
            "row": self.pos.row if self.pos else None,
            "col": self.pos.col if self.pos else None,
            "span": self.pos.span if self.pos else None,
        }

    def print(self, filename: str, file_content: str):
        if self.pos is None:
            print(self.msg, end="", file=sys.stderr)
            sys.stderr.flush()
            global _bcerror_debug
            if _bcerror_debug:
                raise RuntimeError("a traceback is provided:")
            else:
                exit(1)

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

        res = list()

        info = f"{filename}:{line_no}: "
        res.append(f"\033[0m\033[1m{info}")
        msg_lines = self.msg.splitlines()
        res.append(msg_lines[0])  # splitlines on a non-empty string guarantees one elem
        for msg_line in msg_lines[1:]:
            sp = " " * len(info)
            res.append(f"\033[2m\n{sp}{msg_line}\033[0m")
        res.append("\n")

        res.append(line_begin)
        res.append(file_content[bol:eol])
        res.append("\n")

        for ch in file_content[bol:eol]:
            if ch == "\t":
                padding -= 1
                tabs += 1

        tildes = f"{spaces()}\033[31;1m{'~' * self.pos.span}\033[0m"
        res.append(tildes)
        res.append("\n")

        indicator = f"{spaces()}\033[31;1m"
        if os.name == "nt":
            indicator += "+-"
        else:
            indicator += "∟"

        indicator += f" \033[0m\033[1merror at line {line_no} column {col}\033[0m"
        res.append(indicator)

        print("".join(res), file=sys.stdout, flush=True)


def info(msg: str):
    print(
        f"\033[34;1minfo:\033[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def warn(msg: str):
    print(
        f"\033[33;1mwarn:\033[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def error(msg: str):
    print(
        f"\033[31;1merror:\033[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def set_bcerror_debug(state):
    global _bcerror_debug
    _bcerror_debug = state
