from random import choice
import math

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from GameTree import GameTree


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
    _game_tree: GameTree

    _c = 1 # explore-exploit parameter

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self.GameTree = GameTree(Board(self._board_size), self._board_size)

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

        # if turn == 2 and choice([0, 1]) == 1:
        if turn == 2:
            return Move(-1, -1)
        else:
            x, y = choice(self._choices)
            return Move(x, y)
        
    def _select_node(self, game_tree: GameTree) -> GameTree:
        """
        Implements the selection phase of MCTS. 
        Recursivelt selects a node to expand based on the UCB score.

        Args:
            game_tree (GameTree): The current game tree node

        Returns:
            GameTree: The selected node
        """
        for child in game_tree.get_children():
            # calculate UCB score
            ucb_score = (child.q / child.n) + self._c * (math.log(self._game_tree.n) / child.n) ** 0.5
            if ucb_score > max_score:
                max_score = ucb_score
                selected_node = child
        
        return self._select_node(selected_node)
    

    def _expand_node(self, board: Board) -> GameTree:
        node = self._game_tree.get_node(board)
        if node is None:
            raise ValueError("Node not found in game tree")
        
        # choose an unvisited child node
        moves = node.get_valid_moves()
        unvisited_nodes = []
        for move in moves: 
            new_board = board.copy()
            new_board.make_move(move, self.colour)
            child = self._game_tree.get_node(new_board)
            if child is None or child.n == 0:
                unvisited_nodes.append(move)

        if len(unvisited_nodes) == 0:
            raise ValueError("No unvisited nodes found")
        
        selected_child = choice(unvisited_nodes)
        self._game_tree.add_child(selected_child)
        return selected_child

    def _simulate(self, board: Board):
        pass

    def _backup(self, board: Board):
        pass
    
    def _default_policy(self, board: Board) -> Move:
        """
        Default policy for MCTS. Selects a random valid move.

        Args:
            board (Board): The current board state
        
        Returns:
            Move: The agent's move    
        """
        # update valid choices
        node = self._game_tree.get_node(board)
        if node is None:
            raise ValueError("Node not found in game tree")
        self._choices = node.get_valid_moves()
        x, y = choice(self._choices)
        return Move(x, y)