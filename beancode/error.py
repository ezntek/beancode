class BCError(Exception):
    # row, col, bol
    pos: tuple[int, int, int] | None
    eof: bool
    proc: str | None
    func: str | None

    def __init__(self, msg: str, ctx=None, eof=False, proc=None, func=None) -> None:  # type: ignore
        self.eof = eof
        self.len = 1
        self.proc = proc
        self.func = func
        self.ctx = None
        if type(ctx).__name__ == "Token":
            self.pos = ctx.pos  # type: ignore
            self.len = len(ctx.get_raw()[0])  # type: ignore
        elif type(ctx) == tuple:
            self.pos = ctx
        else:
            self.pos = (0, 0, 0)  # type: ignore

        s = f"\033[31;1merror: \033[0m{msg}\n"
        self.msg = s
        super().__init__(s)

    def print(self, filename: str, file_content: str):
        if self.pos is None:
            print(self.msg, end="")
            return

        line = self.pos[0]
        col = self.pos[1]
        bol = self.pos[2]

        try:
            if line != 1 and bol == 0:
                i = 1
                j = -1
                while i < line and j < len(file_content):
                    j += 1
                    while file_content[j] != '\n':
                        j += 1
                    i += 1 
                bol = j + 1
        except IndexError:
            print(self.msg, end="")
            print("\033[33mhint: \033[0mthis probably happened in a procedure or function in the REPL.")
            return

        eol = bol
        while eol != len(file_content) and file_content[eol] != "\n":
            eol += 1

        line_begin = f" \033[31;1m{line}\033[0m | "
        padding = len(str(line) + "  | ") + col - 1
        spaces = lambda *_: " " * padding

        print(f"\033[0m\033[1m{filename}:{line}: ", end="")
        print(self.msg, end="")

        print(line_begin, end="")
        print(file_content[bol:eol])

        tildes = f"{spaces()}\033[31;1m{'~' * self.len}\033[0m"
        print(tildes)
        indicator = f"{spaces()}\033[31;1m∟ \033[0m\033[1merror at line {line} column {col}\033[0m"
        print(indicator)


class BCWarning(Exception):
    # row, col, bol
    pos: tuple[int, int, int]

    def __init__(self, msg: str, ctx=None, data=None) -> None:  # type: ignore
        self.len = 1
        self.data = data
        if type(ctx).__name__ == "Token":
            self.pos = ctx.pos  # type: ignore
            self.len = len(ctx.get_raw()[0])  # type: ignore
        elif type(ctx) == tuple[int, int, int]:
            self.pos = ctx
        else:
            self.pos = (0, 0, 0)  # type: ignore

        s = f"\033[35;1mwarning: \033[0m\033[2m{msg}\033[0m\n"
        self.msg = s
        super().__init__(s)

    def print(self, filename: str, file_content: str):
        line = self.pos[0]
        col = self.pos[1]
        bol = self.pos[2]

        eol = bol
        while eol != len(file_content) and file_content[eol] != "\n":
            eol += 1

        if self.pos == (0, 0, 0):
            print(self.msg, end="")
            return

        line_begin = f" \033[35;1m{line}\033[0m | "
        padding = len(str(line) + "  | ") + col - 1
        spaces = lambda *_: " " * padding

        print(f"\033[0m\033[1m{filename}:{line}: ", end="")
        print(self.msg, end="")

        print(line_begin, end="")
        print(file_content[bol:eol])

        tildes = f"{spaces()}\033[35;1m{'~' * self.len}\033[0m"
        print(tildes)
        indicator = f"{spaces()}\033[35;1m∟ \033[0m\033[1mwarning at line {line} column {col}\033[0m"
        print(indicator)

