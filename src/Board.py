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
        # store coordinates of winning path tiles
        self._winning_path: set[tuple[int, int]] = set()

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

        # reset winner and path before search
        self._winner = None
        self._winning_path.clear()

        # Red
        if colour == Colour.RED:
            for idx in range(self._size):
                tile = self._tiles[0][idx]
                if (
                    not tile.is_visited()
                    and tile.colour == Colour.RED
                    and self._winner is None
                ):
                    self.DFS_colour(0, idx, Colour.RED)

        # Blue
        elif colour == Colour.BLUE:
            for idx in range(self._size):
                tile = self._tiles[idx][0]
                if (
                    not tile.is_visited()
                    and tile.colour == Colour.BLUE
                    and self._winner is None
                ):
                    self.DFS_colour(idx, 0, Colour.BLUE)
        else:
            raise ValueError("Invalid colour")

        # un-visit tiles for later use
        self.clear_tiles()

        # if winner found, compute shortest path
        if self._winner is not None:
            self._compute_shortest_winning_path(colour)
            return True

        return False

    def clear_tiles(self):
        """Clears the visited status from all tiles."""

        for line in self._tiles:
            for tile in line:
                tile.clear_visit()

    def DFS_colour(self, x, y, colour):
        """DFS just to detect a win, not to store the path."""

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
            if 0 <= x_n < self._size and 0 <= y_n < self._size:
                neighbour = self._tiles[x_n][y_n]
                if not neighbour.is_visited() and neighbour.colour == colour:
                    self.DFS_colour(x_n, y_n, colour)

    def print_board(self) -> str:
        size = len(self._tiles)
        output = ""

        # Top red edge (column indices in red)
        output += "  " + "".join(Colour.red(f"{i:2d}")
                                 for i in range(size)) + "\n"

        leading_spaces = ""
        for row_index, line in enumerate(self._tiles):
            # Left blue edge (row index in blue)
            output += " " + leading_spaces + Colour.blue(f"{row_index:2d}")

            for col_index, tile in enumerate(line):
                if (row_index, col_index) in self._winning_path:
                    # raw symbol
                    if tile.colour == Colour.RED:
                        base = "R"
                    elif tile.colour == Colour.BLUE:
                        base = "B"
                    else:
                        base = "·"
                    cell_char = Colour.green(base)
                else:
                    cell_char = Colour.get_char(tile.colour)

                output += cell_char + " "

            output += Colour.blue(f"{row_index:2d}") + "\n"
            leading_spaces += " "

        output += " " + leading_spaces + "".join(
            Colour.red(f"{i:2d}") for i in range(size)
        ) + "\n"

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

    def _compute_shortest_winning_path(self, colour: Colour):
        """Use BFS to find a shortest connection between the two sides
        for the winning colour and store it in _winning_path.
        """

        from collections import deque

        size = self._size
        visited: set[tuple[int, int]] = set()
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        q: deque[tuple[int, int]] = deque()

        # initialise BFS sources depending on colour
        if colour == Colour.RED:
            # top row sources
            for y in range(size):
                if self._tiles[0][y].colour == Colour.RED:
                    q.append((0, y))
                    visited.add((0, y))
                    parent[(0, y)] = None
        elif colour == Colour.BLUE:
            # left column sources
            for x in range(size):
                if self._tiles[x][0].colour == Colour.BLUE:
                    q.append((x, 0))
                    visited.add((x, 0))
                    parent[(x, 0)] = None
        else:
            return

        target: tuple[int, int] | None = None

        while q:
            x, y = q.popleft()

            # goal test
            if colour == Colour.RED and x == size - 1:
                target = (x, y)
                break
            if colour == Colour.BLUE and y == size - 1:
                target = (x, y)
                break

            # explore neighbours
            for idx in range(Tile.NEIGHBOUR_COUNT):
                x_n = x + Tile.I_DISPLACEMENTS[idx]
                y_n = y + Tile.J_DISPLACEMENTS[idx]
                if 0 <= x_n < size and 0 <= y_n < size:
                    if (x_n, y_n) not in visited and \
                       self._tiles[x_n][y_n].colour == colour:
                        visited.add((x_n, y_n))
                        parent[(x_n, y_n)] = (x, y)
                        q.append((x_n, y_n))

        # reconstruct path if target reached
        if target is not None:
            path: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = target
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            self._winning_path = set(path)
        else:
            # fallback: no path found; leave empty
            self._winning_path.clear()


if __name__ == "__main__":
    b = Board.from_string(
        "0R000B00000,0R000000000,0RBB0000000,0R000000000,0R00B000000,"
        + "0R000BB0000,0R0000B0000,0R00000B000,0R000000B00,0R0000000B0,"
        + "0R00000000B",
        bnf=True,
    )
    b.print_board(bnf=False)
    print(b.has_ended(), b.get_winner())
