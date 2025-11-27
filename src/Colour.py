from enum import Enum


# ANSI escape codes for colours
RESET = "\033[0m"
RED_FG = "\033[31m"
BLUE_FG = "\033[34m"
GREEN_FG = "\033[32m"
EMPTY_FG = "\033[90m"


class Colour(Enum):
    """This enum describes the sides in a game of Hex."""

    # RED is vertical, BLUE is horizontal
    RED = 0
    BLUE = 1

    @staticmethod
    def get_char(colour):
        """Returns a coloured single-character representation."""

        if colour == Colour.RED:
            # red 'R'
            return f"{RED_FG}R{RESET}"
        elif colour == Colour.BLUE:
            # blue 'B'
            return f"{BLUE_FG}B{RESET}"
        else:
            # dim dot for empty
            return f"{EMPTY_FG}·{RESET}"

    @staticmethod
    def from_char(c):
        """Returns a colour from its char representations."""

        if c.upper() == "R":
            return Colour.RED
        elif c.upper() == "B":
            return Colour.BLUE
        else:
            return None

    @staticmethod
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
        print(colour, "->", Colour.get_char(colour))
