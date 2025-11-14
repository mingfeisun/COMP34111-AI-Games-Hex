from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    """Represents a player move in a turn of Hex.
    A swap move is when x=-1 and y=-1.
    """

    _x: int = -1
    _y: int = -1

    def is_swap(self) -> bool:
        return self._x == -1 and self._y == -1

    def __str__(self) -> str:
        if self.is_swap():
            return "SWAP()"
        else:
            return f"(x={self._x}, y={self._y})"

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y
