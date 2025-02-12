from typing import NoReturn

def panic(s: str) -> NoReturn:
    print(f"\033[31;1mpanic: \033[0m\033[2m{s}\033[0m")
    exit(1)
    #raise Exception("panicked")
