from src.Colour import Colour
from src.Tile import Tile


class Board:
    """Class that describes the Hex board."""

    _size: int
    _tiles: list[list[Tile]]
    _winner: Colour | None

    def __init__(self, board_size=11):
        self._size = board_size

        self._tiles = []
        for i in range(board_size):
            new_line = []
            for j in range(board_size):
                new_line.append(Tile(i, j))
            self._tiles.append(new_line)

        self._winner = None

    def __str__(self) -> str:
        return self.print_board()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Board):
            return False

        if self._size != value.size:
            return False

        for i in range(self._size):
            for j in range(self._size):
                if self.tiles[i][j].colour != value.tiles[i][j].colour:
                    return False

        return True

    def from_string(string_input, board_size=11):
        """Loads a board from a string representation. If bnf=True, it will
        load a protocol-formatted string. Otherwise, it will load from a
        human-readable-formatted board.
        """

        b = Board(board_size=board_size)

        lines = [line.strip() for line in string_input.split("\n")]
        for i, line in enumerate(lines):
            chars = line.split(" ")
            for j, char in enumerate(chars):
                b.tiles[i][j].colour = Colour.from_char(char)
        return b

    def has_ended(self, colour: Colour = None):
        """Checks if the game has ended. It will attempt to find a red chain
        from top to bottom or a blue chain from left to right of the board.
        """

        # Red
        # for all top tiles, check if they connect to bottom
        if colour == Colour.RED:
            for idx in range(self._size):
                tile = self._tiles[0][idx]
                if not tile.is_visited() and tile.colour == Colour.RED and self._winner is None:
                    self.DFS_colour(0, idx, Colour.RED)
        # Blue
        # for all left tiles, check if they connect to right
        elif colour == Colour.BLUE:
            for idx in range(self._size):
                tile = self._tiles[idx][0]
                if not tile.is_visited() and tile.colour == Colour.BLUE and self._winner is None:
                    self.DFS_colour(idx, 0, Colour.BLUE)
        else:
            raise ValueError("Invalid colour")

        # un-visit tiles
        self.clear_tiles()

        return self._winner is not None

    def clear_tiles(self):
        """Clears the visited status from all tiles."""

        for line in self._tiles:
            for tile in line:
                tile.clear_visit()

    def DFS_colour(self, x, y, colour):
        """A recursive DFS method that iterates through connected same-colour
        tiles until it finds a bottom tile (Red) or a right tile (Blue).
        """

        self._tiles[x][y].visit()

        # win conditions
        if colour == Colour.RED:
            if x == self._size - 1:
                self._winner = colour
        elif colour == Colour.BLUE:
            if y == self._size - 1:
                self._winner = colour
        else:
            return

        # end condition
        if self._winner is not None:
            return

        # visit neighbours
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = x + Tile.I_DISPLACEMENTS[idx]
            y_n = y + Tile.J_DISPLACEMENTS[idx]
            if x_n >= 0 and x_n < self._size and y_n >= 0 and y_n < self._size:
                neighbour = self._tiles[x_n][y_n]
                if not neighbour.is_visited() and neighbour.colour == colour:
                    self.DFS_colour(x_n, y_n, colour)

    def print_board(self) -> str:
        """Returns the coloured string representation of a board.

        Red connects TOP-BOTTOM.
        Blue connects LEFT-RIGHT.
        """

        size = len(self._tiles)
        output = ""

        # Top red edge (column indices in red)
        output += "  " + \
            "".join(Colour.red(f"{i:2d}") for i in range(size)) + "\n"

        leading_spaces = ""
        for row_index, line in enumerate(self._tiles):
            # Left blue edge (row index in blue)
            output += " " + leading_spaces + Colour.blue(f"{row_index:2d}")

            # Cells
            for tile in line:
                output += Colour.get_char(tile.colour) + " "

            # Right blue edge (row index in blue)
            output += Colour.blue(f"{row_index:2d}") + "\n"

            leading_spaces += " "

        # Bottom red edge
        output += " " + leading_spaces + \
            "".join(Colour.red(f"{i:2d}") for i in range(size)) + "\n"

        return output

    def get_winner(self) -> Colour:
        return self._winner

    @property
    def size(self) -> int:
        return self._size

    @property
    def tiles(self) -> list[list[Tile]]:
        return self._tiles

    def set_tile_colour(self, x, y, colour) -> None:
        self.tiles[x][y].colour = colour


if __name__ == "__main__":
    b = Board.from_string(
        "0R000B00000,0R000000000,0RBB0000000,0R000000000,0R00B000000,"
        + "0R000BB0000,0R0000B0000,0R00000B000,0R000000B00,0R0000000B0,"
        + "0R00000000B",
        bnf=True,
    )
    b.print_board(bnf=False)
    print(b.has_ended(), b.get_winner())
