from enum import Enum

# ANSI escape codes for colours (screen output only, never the protocol)
RESET = "\033[0m"
RED_FG = "\033[31m"
BLUE_FG = "\033[34m"
GREEN_FG = "\033[32m"
EMPTY_FG = "\033[90m"


class _ClassOrInstanceMethod:
    """Lets a method be called as Colour.f(c) and as c.f().

    Accepts the form c.f(other) as well, for compatibility with the 2025 engine.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        func = self.func

        def bound(*args):
            # c.f() uses c; c.f(other) uses other, as a static method called
            # through an instance did on the 2025 engine
            return func(*args) if args else func(obj)

        bound.__doc__ = func.__doc__
        return bound


class Colour(Enum):
    """This enum describes the sides in a game of Hex."""

    # RED is vertical, BLUE is horizontal
    RED = 0
    BLUE = 1

    @_ClassOrInstanceMethod
    def get_char(colour):
        """Returns "R", "B", or "0" for an empty tile (None).

        Same result as to_protocol_char(). Use display_char() for coloured
        screen output.
        """

        if colour == Colour.RED:
            return "R"
        elif colour == Colour.BLUE:
            return "B"
        else:
            return "0"

    @_ClassOrInstanceMethod
    def to_protocol_char(colour):
        """Returns the character used in the external-agent protocol:
        "R", "B", or "0" for an empty tile (None)."""

        return Colour.get_char(colour)

    @_ClassOrInstanceMethod
    def display_char(colour):
        """Returns a coloured single character for screen output only."""

        if colour == Colour.RED:
            return f"{RED_FG}R{RESET}"
        elif colour == Colour.BLUE:
            return f"{BLUE_FG}B{RESET}"
        else:
            return f"{EMPTY_FG}·{RESET}"

    @_ClassOrInstanceMethod
    def from_char(c):
        """Returns a colour from its char representations."""

        if c.upper() == "R":
            return Colour.RED
        elif c.upper() == "B":
            return Colour.BLUE
        else:
            return None

    @_ClassOrInstanceMethod
    def opposite(colour):
        """Returns the opposite colour."""

        if colour == Colour.RED:
            return Colour.BLUE
        elif colour == Colour.BLUE:
            return Colour.RED
        else:
            return None

    @staticmethod
    def red(text: str) -> str:
        return f"{RED_FG}{text}{RESET}"

    @staticmethod
    def blue(text: str) -> str:
        return f"{BLUE_FG}{text}{RESET}"

    @staticmethod
    def green(text: str) -> str:
        return f"{GREEN_FG}{text}{RESET}"


if __name__ == "__main__":
    for colour in Colour:
        print(colour, "->", Colour.get_char(colour), colour.get_char(), Colour.display_char(colour))
