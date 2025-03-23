class BCError(Exception):
    # row, col, bol
    pos: tuple[int, int, int]

    def __init__(self, msg: str, tok = None) -> None: # type: ignore
        if type(tok).__name__ == "Token":
            self.pos = tok.pos # type: ignore
            self.len = len(tok.get()[0]) # type: ignore
        elif type(tok) == tuple:
            self.pos = tok
            self.len = 1
        else:
            self.pos = (0, 0) # type: ignore
            self.len = 1

        s = f"\033[31;1merror: \033[0m\033[2m{msg}\033[0m\n"
        self.msg = s
        super().__init__(s)

    def print(self, filename: str, file_content: str):
        line = self.pos[0]
        col = self.pos[1]
        bol = self.pos[2] 
        eol = bol
        while file_content[eol] != '\n':
            eol += 1
        line_begin = f"    \033[31;1m{line}\033[0m | "
        
        spaces = lambda *_: ' ' * (len(line_begin) + col)
 
        print(f"\033[1mfile \"{filename}\" line {line}: ", end='')
        print(self.msg)

        print(line_begin, end='')
        print(file_content[bol:eol])

        tildes = f"{spaces()}\033[31;1m{'~' * self.len}\033[0m" 
        print(tildes)
        indicator = f"{spaces()}\033[31;1m∟ \033[0m\033[1merror at column {col}\033[0m"
        print(indicator)

class BCParseError(BCError):
    pass

class BCRuntimeError(BCError):
    pass

