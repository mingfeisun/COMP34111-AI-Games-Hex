from random import choice
import math

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

C = 2

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

class GameTree:
    def __init__(self, board: Board, colour: Colour, move: Move = None, parent = None):
        self.board = board
        self.colour = colour
        self.children = []
        self.move = move # represents the move that led to this state
        self.parent = parent

        self.num_visits = 0
        self.value = 0

    def add_move(self, move):
        new_colour = Colour.opposite(self.colour)

        new_board = Board(self.board.size)
        for i in range(self.board.size):
            for j in range(self.board.size):
                new_board.set_tile_colour(i, j, self.board.tiles[i][j].colour)
        new_board.set_tile_colour(move.x, move.y, self.colour)

        new_state = GameTree(new_board, new_colour, move)
        self.children.append(new_state)
        return new_state

    def get_node(self, board):
        for child in self.children:
            if child.board == board:
                return child
        return None

class MCTSAgent(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11
    _first_move: bool = True

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
    
    def uct(self, node: GameTree) -> float:
        parent_visits = node.parent.num_visits
        child_visits = node.num_visits

        if child_visits == 0:
            return math.inf

        child_value = node.value
        return child_value / child_visits + C * (2 * math.ln(parent_visits) / child_visits) ** 0.5

    def select_child_node(self, current: GameTree) -> GameTree:
        if current.board.has_ended(Colour.RED):
            return current
        if current.board.has_ended(Colour.BLUE):
            return current
        if len(current.children) == 0:
            return current

        best_child = current.children[0]
        for child in current.children[1:]:
            if self.uct(child) > self.uct(best_child):
                best_child = child

        return self.select_child_node(best_child)

    def rollout(self, current: GameTree):
        choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
            if current.board.tiles[i][j].colour is None
        ]
        while not current.board.has_ended(Colour.RED) and not current.board.has_ended(Colour.BLUE):
            x, y = choice(self._choices)
            current  = current.add_move(Move(x, y))
        if current.board.get_winner() == self.colour:
            return 1
        return 0
    
    def expand(self, current: GameTree):
        for i, j in self._choices:
            current.add_move(Move(i, j))
        return current
    
    def backpropagate(self, current: GameTree, value: int):
        current.num_visits += 1
        current.value += value
        if current.parent is not None:
            self.backpropagate(current.parent, value)

    def mcts(self, current: GameTree):
        for _ in range(100): # TODO: Change this to execute while time permits
            # update available choices
            self._choices = [
                (i, j) for i in range(self._board_size) for j in range(self._board_size)
                if current.board.tiles[i][j].colour is None
            ]

            # selection
            current = self.select_child_node(current)

            # expansion and rollout
            if current.num_visits == 0:
                value = self.rollout(current)
            else:
                current = self.expand(current)
                if len(current.children) == 0:
                    value = self.rollout(current)
                else:
                    current = current.children[0]
                    value = self.rollout(current)
            
            self.backpropagate(current, value)
    
    def get_best_move(self, current: GameTree) -> Move:
        best_child = current.children[0]
        for child in current.children[1:]:
            avg_value = child.value / child.num_visits
            if avg_value > (best_child.value / best_child.num_visits):
                best_child = child
        return best_child.move
    
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        # initialise the game tree with appropriate colours
        if turn == 1:
            self._game_tree = GameTree(board, self.colour)
        elif turn == 2:
            self._game_tree = GameTree(board, Colour.opposite(self.colour))
            self._game_tree.add_move(opp_move)
        
        self.mcts(self._game_tree)

        # return the best move
        self.get_best_move(self._game_tree)
