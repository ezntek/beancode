from io import StringIO
from typing import Callable

from .libroutine_defs import *

# multiline string literals are cringe and kinda unreadable.
# We use StringIOs here!

HelpEntry = tuple[str, Callable[[], str]]

def _help() -> str:
    res = StringIO()
    print("Show this help message, and a list of available help pages.\n", file=res)
    print("Available help entries include:", file=res)
    for (key, val) in HELP_ENTRIES.items():
        print(f"  - \033[32m{key}\033[0m: {val[0]}", file=res)
    print("\nType help(\"your help entry\") to get more information.")
    return res.getvalue()


def _libroutines() -> str:
    res = StringIO()

    print()

    return res.getvalue()


def _ucase() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _lcase() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _div() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _mod() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _substring() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _round() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _sqrt() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _length() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _sin() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _cos() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _tan() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _getchar() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _random() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _putchar() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _exit() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


def _sleep() -> str:
    res = StringIO()

# TODO: write help

    return res.getvalue()


def _flush() -> str:
    res = StringIO()

    # TODO: write help

    return res.getvalue()


_bcext = "\033[2m[beancode extension] \033[0m"

LIBROUTINE_ENTRIES = {
    "ucase": ("Get the uppercase value of a string.", _ucase),
    "lcase": ("Get the lowercase value of a string.", _lcase),
    "div": ("Floor-divide a numeric value by a numeric value.", _div),
    "mod": ("Find the remainder of a numeric value when divided by a numeric value.", _mod),
    "substring": ("Extract a sub-string from a string.", _substring),
    "round": ("Round a value to a certain number of decimal places.", _round),
    "length": ("Find the length of a string.", _length),
    "random": ("Get a random value between 0 and 1, inclusive.", _random),
    "sqrt": (f"{_bcext} Find the square root of a value.", _sqrt),
    "sin": (f"{_bcext} Find the sine of a value in radians.", _sin),
    "cos": (f"{_bcext} Find the cosine of a value in radians.", _cos),
    "tan": (f"{_bcext} Find the tangent of a value in radians.", _tan),
    "getchar": (f"{_bcext} Get a single character from the standard input.", _getchar),
    "putchar": (f"{_bcext} Print a single character to the standard output.", _putchar),
    "exit": (f"{_bcext} Exit from the running program with an exit code.", _exit),
    "sleep": (f"{_bcext} Sleep for n seconds.", _sleep),
    "flush": (f"{_bcext} Flush the standard output.", _flush),
}

HELP_ENTRIES: dict[str, HelpEntry] = {
    "help": ("Show some information regarding the help library routine.", _help),
    "library routines": ("Information regarding library routines", _libroutines),
    # library routines
}
HELP_ENTRIES.update(LIBROUTINE_ENTRIES)

def bean_help(query: str) -> str | None:
    entry = HELP_ENTRIES.get(query.lower())
    if entry is None:
        return None
    return f"\033[32;1m=== beancode help for \033[0m{query}\033[32;1m ===\033[0m\n" + entry[1]() 
