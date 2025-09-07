__version__ = "0.3.0-alpha4"


def error(msg: str):
    print(f"\033[31;1merror: \033[0m{msg}")
    exit(1)


def panic(msg: str):
    print(f"\033[31;1mpanic! \033[0m{msg}")
    print(
        "\033[31mplease report this error to the developers. A traceback is provided:\033[0m"
    )
    raise Exception("panicked")
