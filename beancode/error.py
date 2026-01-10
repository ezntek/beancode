# beancode: a portable IGCSE Computer Science (0478, 0984, 2210) Pseudocode interpreter.
#
# Copyright (c) Eason Qin, 2025-2026.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

import sys

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

        self.msg = msg
        super().__init__(msg)

    def to_dict(self, file_content: str | None = None) -> dict[str, str | int | None]:
        frm = None
        to = None

        if self.pos and file_content:
            bol, eol = self._get_line_start_end(self.pos.row, file_content)
            frm = bol + self.pos.col - 1  # col 1 = bol + 0
            to = frm + self.pos.span
            if to > eol:
                to = eol

        return {
            "msg": self.msg,
            "from": frm if frm else None,
            "to": to if to else None,
        }

    @staticmethod
    def _get_line_start_end(line_no: int, file_content: str):
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

        return (bol, eol)

    def print_compact(self, pos: Pos, filename: str, file_content: str):
        _ = file_content
        line_no = pos.row
        col = pos.col
        res = list()

        line_begin = f" \x1b[31;1m{line_no}\x1b[0m | "
        bol, eol = self._get_line_start_end(line_no, file_content)
        snippet = file_content[bol:eol]
        begin_space_count = 0
        for ch in snippet:
            if not ch.isspace():
                break

            if ch in "\t ":
                begin_space_count += 1

        info = (
            f"\x1b[1m{filename}: \x1b[31merror\x1b[0m at line {line_no} column {col}:"
        )
        res.append(info + "\n")
        res += self.msg
        res.append("\n")


        res.append(line_begin)
        res.append(snippet.strip())
        res.append("\n")

        # 4: space, <number>, space, pipe, space
        #    ^^^^^            ^^^^^  ^^^^  ^^^^^
        padding = (col - begin_space_count) + len(str(line_no)) + 3

        tildes = f"{' ' * padding}\x1b[31;1m{'~' * pos.span}\x1b[0m"
        res.append(tildes)

        print("".join(res), file=sys.stdout, flush=True)

    def print_normal(self, pos: Pos, filename: str, file_content: str):
        line_no = pos.row
        col = pos.col
        bol, eol = self._get_line_start_end(line_no, file_content)

        line_begin = f" \x1b[31;1m{line_no}\x1b[0m | "
        padding = len(str(line_no) + "  | ") + col - 1
        tabs = 0
        spaces = lambda *_: " " * padding + "\t" * tabs

        res = list()

        info = f"{filename}:{line_no}: "
        res.append(f"\x1b[0m\x1b[1m{info}")
        msg_lines = ("\x1b[31;1merror: \x1b[0m" + self.msg).splitlines()
        res.append(msg_lines[0])  # splitlines on a non-empty string guarantees one elem
        for msg_line in msg_lines[1:]:
            sp = " " * len(info)
            res.append(f"\x1b[2m\n{sp}{msg_line}\x1b[0m")
        res.append("\n")

        res.append(line_begin)
        res.append(file_content[bol:eol])
        res.append("\n")

        for ch in file_content[bol:eol]:
            if ch == "\t":
                padding -= 1
                tabs += 1

        tildes = f"{spaces()}\x1b[31;1m{'~' * pos.span}\x1b[0m"
        res.append(tildes)
        res.append("\n")

        indicator = f"{spaces()}\x1b[31;1m"
        if sys.platform == "nt":
            indicator += "+-"
        else:
            indicator += "∟"

        indicator += f" \x1b[0m\x1b[1merror at line {line_no} column {col}\x1b[0m"
        res.append(indicator)

        print("".join(res), file=sys.stdout, flush=True)

    def print(self, filename: str, file_content: str, compact=False):
        try:
            if self.pos is None:
                print("\x1b[31;1merror: \x1b[0m" + self.msg, end="\n", file=sys.stderr)
                sys.stderr.flush()
                global _bcerror_debug
                if _bcerror_debug:
                    raise RuntimeError("a traceback is provided:")
                else:
                    return

            if compact:
                self.print_compact(self.pos, filename, file_content)
            else:
                self.print_normal(self.pos, filename, file_content)
        except KeyboardInterrupt:
            return


def info(msg: str):
    print(
        f"\x1b[34;1minfo:\x1b[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def warn(msg: str):
    print(
        f"\x1b[33;1mwarn:\x1b[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def error(msg: str):
    print(
        f"\x1b[31;1merror:\x1b[0m {msg}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def set_bcerror_debug(state):
    global _bcerror_debug
    _bcerror_debug = state
