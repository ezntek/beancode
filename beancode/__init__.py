__version__ = "0.1.1"

class BCError(Exception):
    # row, col, bol
    pos: tuple[int, int, int]

    def __init__(self, msg: str, ctx=None) -> None:  # type: ignore
        self.len = 1
        if type(ctx).__name__ == "Token":
            self.pos = ctx.pos  # type: ignore
            self.len = len(ctx.get_raw()[0])  # type: ignore
        elif type(ctx) == tuple:
            self.pos = ctx
        else:
            self.pos = (0, 0, 0)  # type: ignore

        s = f"\033[31;1merror: \033[0m\033[2m{msg}\033[0m\n"
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
        padding = len(str(line) + "  | ") + col
        spaces = lambda *_: " " * padding

        print(f"\033[0m\033[1m{filename}:{line}: ", end="")
        print(self.msg, end="")

        print(line_begin, end="")
        print(file_content[bol:eol])

        tildes = f"{spaces()}\033[35;1m{'~' * self.len}\033[0m"
        print(tildes)
        indicator = f"{spaces()}\033[35;1m∟ \033[0m\033[1mwarning at line {line} column {col}\033[0m"
        print(indicator)


def error(msg: str):
    print(f"\033[31;1merror: \033[0m{msg}")
    exit(1)


def panic(msg: str):
    print(f"\033[31;1mpanic! \033[0m{msg}")
    print(
        "\033[31mplease report this error to the developers. A traceback is provided:\033[0m"
    )
    raise Exception("panicked")
