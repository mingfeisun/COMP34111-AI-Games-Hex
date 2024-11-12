import math

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
import random
import time
import numpy as np
from collections import defaultdict
import copy

def has_game_ended(state: Board) -> bool:
    """
    Check if the game has ended
    """
    return copy.deepcopy(state).has_ended(Colour.RED) or copy.deepcopy(state).has_ended(Colour.BLUE)

def get_valid_moves(state: Board) -> list[tuple]:
    """
    Get all valid moves for the given state
    """
    return [(i, j) for i in range(state.size) for j in range(state.size) if state.tiles[i][j].colour is None]

def get_opponent_colour(colour: Colour) -> Colour:
    """
    Get the opponent colour
    """
    return Colour.RED if colour == Colour.BLUE else Colour.BLUE

class MCTSNode:
    """
    Node for MCTS
    """
    def __init__(self, state: Board, parent=None, move=None):
        """
        Initialize the node
        """
        self.state: Board = state
        self.parent: MCTSNode = parent
        self.children: list[MCTSNode] = []
        self.move: tuple = move
        self.visits: int = 0
        self.value: int = 0
        self.valid_moves: list[tuple] = get_valid_moves(state)
        self.unexplored_moves: list[tuple] = self.valid_moves

    def is_fully_expanded(self) -> bool:
        """
        Check if the node is fully expanded
        That is there are no more unexplored moves

        Returns true if all possible moves have been expanded
        """
        return len(self.unexplored_moves) == 0

    def expand(self):
        """
        Expand the node by adding the children nodes
        """
        if not self.unexplored_moves:
            return None
        move = self.unexplored_moves[np.random.randint(len(self.unexplored_moves))]
        new_state = copy.deepcopy(self.state)
        new_state.set_tile_colour(move[0], move[1], Colour.RED if len(self.children) % 2 == 0 else Colour.BLUE)

        child_node = MCTSNode(state=new_state, parent=self, move=move)
        self.children.append(child_node)
        self.unexplored_moves.remove(move)

        return child_node


    def best_child(self, c: float):
        """
        Select the best child node based on UCT

        Formula:
            w_i / n_i + c * sqrt(t) / n_i
            w_i: number of wins for the node
            n_i: number of simulations for the node
            c: exploration parameter
            t: total number of simulations
        """
        values = np.array([child.value for child in self.children])
        visits = np.array([child.visits for child in self.children])

        exploitation = np.where(visits > 0, values / visits, 0)
        total_visits_log = np.log(self.visits)
        exploration = np.where(visits > 0, c * np.sqrt(total_visits_log / visits), float('inf'))
        ucb1_values = exploitation + c * exploration
        best_index = np.argmax(ucb1_values)
        return self.children[best_index]


class MCTSAgent(AgentBase):
    """
    Monte Carlo Tree Search Agent
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def select(self, node: MCTSNode):
        """
        Select the best child node based on UCT
        """
        while not node.is_fully_expanded():
            node = node.best_child(c=1.41)
        return node


    def simulate(self, node: MCTSNode):
        """
        Simulate the game until the end with random moves
        """
        current_state = copy.deepcopy(node.state)
        while not has_game_ended(current_state):
            move = random.choice(self._choices)
            current_state.tiles[move[0]][move[1]].colour = self.colour
            if has_game_ended(copy.deepcopy(current_state)):
                break
            move = random.choice(self._choices)
            current_state.tiles[move[0]][move[1]].colour = get_opponent_colour(self.colour)
            if has_game_ended(copy.deepcopy(current_state)):
                break
        copy1 = copy.deepcopy(current_state)
        copy2 = copy.deepcopy(current_state)
        if copy1.has_ended(self.colour):
            return copy1.get_winner()
        elif copy2.has_ended(get_opponent_colour(self.colour)):
            return copy2.get_winner()

    def backpropagate(self, node: MCTSNode, result: int):
        """
        Update the nodes in the tree with the result of the simulation
        """
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Make move based on MCTS
        """
        if turn == 2:
            # This is the first move of the game
            # Can swap if we want
            if bool(random.getrandbits(1)):
                return Move(-1, -1)
        # Return the best move by MCTS
        root_node = MCTSNode(board)
        while not root_node.is_fully_expanded():
            root_node.expand()

        for i in range(10):
            node = self.select(root_node)
            if not node.is_fully_expanded():
                node.expand()
            result = self.simulate(node)
            self.backpropagate(node, 1 if result == self.colour else -1)
        move_tuple = root_node.best_child(1.42).move
        return Move(move_tuple[0], move_tuple[1])


