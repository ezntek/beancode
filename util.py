from typing import NoReturn

def panic(s: str) -> NoReturn:
    print(s)
    exit(1)
