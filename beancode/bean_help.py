from io import StringIO
from typing import Callable

# multiline string literals are cringe and kinda unreadable.
# We use StringIOs here!

HelpEntry = tuple[str, Callable[[], str]]


def _help() -> str:
    res = StringIO()
    print("Show this help message, and a list of available help pages.\n", file=res)
    print("Available help entries include:", file=res)
    for key, val in HELP_ENTRIES.items():
        print(f"  - \033[32m{key}\033[0m: {val[0]}", file=res)
    print('\nType help("your help entry") to get more information.', file=res)
    return res.getvalue()


def _libroutines() -> str:
    res = StringIO()

    print(
        "Library routines are built-in features of Pseudocode that simplify tasks for you.",
        file=res,
    )
    print(
        """
There are many of them; some are marked with \033[2m[beancode extension]\033[0m, which means
that it is not an \"official\" library routine specified by Cambridge, but rather
a custom one that beancode includes to further simplify programming.
    """,
        file=res,
    )

    print("Here are the ones that Beancode supports:", file=res)
    for key, val in LIBROUTINE_ENTRIES.items():
        print(f"  - \033[32m{key}\033[0m: {val[0]}", file=res)
    print('\nType help("your library routine") to get more information.', file=res)

    return res.getvalue()


def _ucase() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["ucase"][0] + "\n")
    res.write("Arguments: (STRING or CHAR), Returns: STRING or CHAR\n\n")
    res.write(
        """Examples:
  UCASE("hello") \033[2m// returns "HELLO"\033[0m
  UCASE("Srinivasa Ramanujan") \033[2m// returns "SRINIVASA RAMANUJAN"\033[0m
    """
    )

    return res.getvalue()


def _lcase() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["lcase"][0] + "\n")
    res.write("Arguments: (STRING), Returns: STRING\n\n")
    res.write(
        """Examples:
  LCASE("GOODBYE") \033[2m// returns "goodbye"\033[0m
  LCASE("Alexey Kutepov") \033[2m// returns "alexey kutepov"\033[0m
    """
    )

    return res.getvalue()


def _div() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["div"][0] + "\n")
    res.write("Arguments: (REAL or INTEGER), Returns: REAL or INTEGER\n\n")
    res.write(
        """Examples:
  DIV(5, 2) \033[2m// returns 2\033[0m
  DIV(12.5, 2) \033[2m// returns 6\033[0m
    """
    )

    return res.getvalue()


def _mod() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["mod"][0] + "\n")
    res.write("Arguments: (REAL or INTEGER), Returns: REAL or INTEGER\n\n")
    res.write(
        """Examples:
  MOD(5, 2) \033[2m// returns 1\033[0m
  MOD(7.3, 4.1) \033[2m// returns 3.2\033[0m
    """
    )

    return res.getvalue()


def _substring() -> str:
    res = StringIO()

    # TODO: write help
    res.write(
        f"""{LIBROUTINE_ENTRIES["substring"][0]} 
SUBSTRING works on the base string, where you want to begin taking characters
and how many characters to take in total. \033[31;1mIn beancode, if you call SUBSTRING()
with length 1, it will return a CHAR.\033[0m\n
\033[1mNote: IGCSE Pseudocode strings start at 1.\n\033[0m
"""
    )
    res.write("Arguments: (STRING, INTEGER, INTEGER), Returns: STRING or CHAR\n")
    res.write(
        """Example:
  DECLARE Original, First, Second: STRING
  DECLARE Letter: CHAR
  Original <- "Fish and Chips"

  First <- SUBSTRING(Original, 1, 4) \033[2m// stores "Fish" into First\033[0m
  Second <- SUBSTRING(Original, 10, 5) \033[2m// stores "Chips" into Second\033[0m
  Letter <- SUBSTRING(Original, 3, 1) \033[2m// stores 's' into Letter as a CHAR\033[0m
"""
    )

    return res.getvalue()


def _length() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["length"][0] + "\n")
    res.write("Arguments: (STRING), Returns: INTEGER\n")
    res.write(
        """Examples:
  LENGTH("Andrew Kelley") \033[2m// returns 13\033[0m
  LENGTH("Drew DeVault") \033[2m// returns 12\033[0m
    """
    )

    return res.getvalue()


def _round() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["round"][0] + "\n")
    res.write("Arguments: (REAL or INTEGER), Returns: REAL or INTEGER\n")
    res.write(
        """Examples:
  ROUND(3.1415926, 2) \033[2m// returns 3.14\033[0m
  ROUND(5.2613, 0) \033[2m// returns 5 as an integer\033[0m
    """
    )

    return res.getvalue()


def _random() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["random"][0] + "\n")
    res.write("Arguments: (), Returns: REAL\n")

    return res.getvalue()


EXTENSION_TXT = """\033[31mThis is a library routine that is part of the beancode extensions.\033[0m
This means that it is not found in standard IGCSE Pseudocode, but is specific
to beancode. If you use this function on an exam, the examiner will know what
you mean, but keep in mind that it is not in the IGCSE Pseudocode standard!
    \n"""


def _sqrt() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["sqrt"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (REAL or INTEGER), Returns: REAL\n")

    return res.getvalue()


def _sin() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["sin"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (REAL), Returns: REAL\n")

    return res.getvalue()


def _cos() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["cos"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (REAL), Returns: REAL\n")

    return res.getvalue()


def _tan() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["tan"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (REAL), Returns: REAL\n")

    return res.getvalue()


def _getchar() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["getchar"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (), Returns: CHAR\n")

    return res.getvalue()


def _putchar() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["putchar"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (CHAR)")

    return res.getvalue()


def _exit() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["exit"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (INTEGER)")

    return res.getvalue()


def _sleep() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["sleep"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (INTEGER or REAL)")
    return res.getvalue()


def _flush() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["flush"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: ()")

    return res.getvalue()


def _execute() -> str:
    res = StringIO()

    res.write(LIBROUTINE_ENTRIES["execute"][0] + "\n")
    res.write(EXTENSION_TXT)
    res.write("Arguments: (STRING), Returns: STRING")

    return res.getvalue()


_bcext = "\033[2m[beancode extension]\033[0m"

LIBROUTINE_ENTRIES = {
    "ucase": ("Get the uppercase value of a string.", _ucase),
    "lcase": ("Get the lowercase value of a string.", _lcase),
    "div": ("Floor-divide a numeric value by a numeric value.", _div),
    "mod": (
        "Find the remainder of a numeric value when divided by a numeric value.",
        _mod,
    ),
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
    "execute": (f"{_bcext} Execute a system command in a shell.", _execute),
}

HELP_ENTRIES: dict[str, HelpEntry] = {
    "help": ("Show some information regarding the help library routine.", _help),
    "library routines": ("Information regarding library routines", _libroutines),
    "libroutines": ("Information regarding library routines", _libroutines),
    # library routines
}
HELP_ENTRIES.update(LIBROUTINE_ENTRIES)


def bean_help(query: str) -> str | None:
    entry = HELP_ENTRIES.get(query.lower())
    if entry is None:
        return None
    return (
        f"\033[32;1m=== beancode help for \033[0m{query}\033[32;1m ===\033[0m\n"
        + entry[1]()
    )
