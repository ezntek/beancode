from beancode.bean_ast import BCType

Libroutine = list[tuple[BCType, ...] | BCType]
Libroutines = dict[str, Libroutine]

LIBROUTINES: Libroutines = {
    "ucase": ["string"],
    "lcase": ["string"],
    "div": [("integer", "real"), ("integer", "real")],
    "mod": [("integer", "real"), ("integer", "real")],
    "substring": ["string", "integer", "integer"],
    "round": ["real", "integer"],
    "sqrt": [("integer", "real")],
    "length": ["string"],
    "sin": ["real"],
    "cos": ["real"],
    "tan": ["real"],
    "help": ["string"],
    "getchar": [],
    "random": [],
}

LIBROUTINES_NORETURN: Libroutines = {
    "putchar": ["char"],
    "exit": ["integer"],
    "sleep": ["real"],
    "flush": [],
}
