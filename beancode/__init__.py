from dataclasses import dataclass

__version__ = "0.5.0-dev"


@dataclass
class Pos:
    row: int
    col: int
    span: int

    def __repr__(self) -> str:
        return f"{self.row} {self.col} {self.span}"


def prefix_string_with_article(s: str) -> str:
    if s[0].lower() in "aeiou":
        return "an " + s
    else:
        return "a " + s


def humanize_index(idx: int) -> str:
    s = str(idx)
    last = s[-1]

    if len(s) == 1 or (len(s) > 1 and s[-2] != "1"):
        match last:
            case "1":
                return s + "st"
            case "2":
                return s + "nd"
            case "3":
                return s + "rd"
            case _:
                return s + "th"

    return s + "th"


def is_case_consistent(s: str) -> bool:
    return s.isupper() or s.islower()

    
def is_integer(val: str) -> bool:
    if len(val) == 0:
        return False

    for ch in val:
        if not ch.isdigit():
            return False
    return True

def is_real(val: str) -> bool:
    if len(val) == 0:
        return False

    if is_integer(val):
        return False

    found_decimal = False

    for ch in val:
        if ch == ".":
            if found_decimal:
                return False
            found_decimal = True

    return found_decimal


def error(msg: str):
    print(f"\033[31;1merror: \033[0m{msg}")
    exit(1)


def panic(msg: str):
    print(f"\033[31;1mpanic! \033[0m{msg}")
    print(
        "\033[31mplease report this error to the developers. A traceback is provided:\033[0m"
    )
    raise Exception("panicked")
