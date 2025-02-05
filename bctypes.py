from dataclasses import dataclass
from util import *
import typing

BCType = typing.Literal["integer", "real", "char", "string", "boolean"]


@dataclass
class BCValue:
    kind: BCType
    integer: int | None = None
    real: float | None = None
    char: str | None = None
    string: str | None = None
    boolean: bool | None = None

    def get_integer(self) -> int:
        if self.kind != "integer":
            panic("incorrect type")

        return self.integer  # type: ignore

    def get_real(self) -> float:
        if self.kind != "real":
            panic("incorrect type")

        return self.real  # type: ignore

    def get_char(self) -> str:
        if self.kind != "char":
            panic("incorrect type")

        return self.char  # type: ignore

    def get_string(self) -> str:
        if self.kind != "string":
            panic("incorrect type")

        return self.string  # type: ignore

    def get_boolean(self) -> bool:
        if self.kind != "boolean":
            panic("incorrect type")

        return self.boolean  # type: ignore

    def __repr__(self) -> str:
        match self.kind:
            case "string":
                return self.get_string()
            case "real":
                return str(self.get_real())
            case "integer":
                return str(self.get_integer())
            case "char":
                return str(self.get_char())
            case "boolean":
                return str(self.get_boolean())
