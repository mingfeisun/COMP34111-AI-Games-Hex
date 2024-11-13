import random

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
import copy
import numpy as np

def get_valid_moves(state: Board) -> list[tuple]:
    """
    Get all valid moves for the given state
    """
    return [(i, j) for i in range(state.size) for j in range(state.size) if state.tiles[i][j].colour is None]

def get_opponent_colour(colour: Colour) -> Colour:
    """
    Get the opponent colour
    """
    if (colour != Colour.RED) and (colour != Colour.BLUE):
        raise ValueError("Invalid colour")
    return Colour.RED if colour == Colour.BLUE else Colour.BLUE

class MCTSNode:
    """
    Node for MCTS
    """
    def __init__(self, state: Board, parent=None, move=None):
        """
        Initialize the node
        Arguments:
            state: State of the board in this node
            parent: Parent node
            move: The move that was made to reach this node
        """
        self.state: Board = state
        self.parent: MCTSNode = parent
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.payoff_sum: int = 0
        # Use tuples for the move to avoid overhead of using Move objects
        # In the form (x, y)
        self.move: tuple = move
        self.valid_moves: list[tuple] = get_valid_moves(state)

    def generate_all_children_nodes(self, player_colour: Colour):
        """
        Generate all children nodes
        Input:
            player_colour: Colour of the player
        """
        # Create all children nodes with the same state as the parent
        self.children = [
            MCTSNode(
                state=copy.deepcopy(self.state),  # Deepcopy here is unavoidable
                parent=self,
                move=move,
            ) for move in self.valid_moves
        ]

        # Apply the move to each of the children nodes
        for child, move in zip(self.children, self.valid_moves):
            child.state.tiles[move[0]][move[1]].colour = player_colour

    def backpropagate(self, result: int):
        """
        Update the nodes in the tree with the result of the simulation
        Arguments:
            result: 1 if the player won, -1 if the opponent won
        """
        node = self
        while node is not None:
            node.visits += 1
            node.payoff_sum += result
            node = node.parent

    def simulate_from_node(self, current_colour: Colour):
        """Simulate the game until the end with random moves

        Arguments:
            current_colour: Colour of the current player
        """
        moves_taken = []
        valid_moves = get_valid_moves(self.state)
        if self.state.has_ended(get_opponent_colour(current_colour)):
            # If the current state is a winning state
            # Return the winner
            winner = self.state.get_winner()
            self.state._winner = None
            return winner
        while True:
            # Faster to remove the move from the list than to generate a new list every move
            move = random.choice(valid_moves)
            valid_moves.remove(move)
            moves_taken.append(move)

            # Do the move
            self.state.tiles[move[0]][move[1]].colour = current_colour

            # Check if the game has ended
            if self.state.has_ended(current_colour):
                # Hacky way to reset the board state
                winner = self.state.get_winner()
                self.state._winner = None  # Reset the winner technically _winner is private but python allows us to get around this
                for move in moves_taken:
                    self.state.tiles[move[0]][move[1]].colour = None
                return winner

            # Switch the player for the next move
            current_colour = get_opponent_colour(current_colour)

    def best_child(self, c: float):
        """
        Select the best child node based on UCT
        Arguments:
            c: Exploration parameter
        """
        values = np.array([child.payoff_sum for child in self.children])
        visits = np.array([child.visits for child in self.children])

        # If there are unexplored nodes we will always choose one of them so don't bother with UCT
        unexplored = np.where(visits == 0)[0]
        if unexplored.size > 0:
            best_index = unexplored[np.random.randint(0, unexplored.size)]
            return self.children[best_index]

        # Calculate the UCT value for each child and select the best one
        exploitation = values / visits
        exploration = c * np.sqrt(np.log(self.visits) / visits)
        ucb1_values = exploitation + exploration
        return self.children[np.argmax(ucb1_values)]


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


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Make move based on MCTS
        Arguments:
            turn: Current turn number
            board: Current state of the board
            opp_move: Move made by the opponent
        """
        # At the moment always switch on the second turn
        # TODO: Implement a better way to decide if we should swap
        if turn == 2:
            return Move(-1, -1)

        root = MCTSNode(state=board)
        root.generate_all_children_nodes(self.colour)

        # Should use some time limit here based on how much time we have left
        for _ in range(10000):
            # Use the tree policy to select the best node
            # Uses UCT to select the best node
            child_to_expand = root.best_child(c=1.41)
            # Simulate the game until the end
            result_colour = child_to_expand.simulate_from_node(get_opponent_colour(self.colour))
            # Backpropagate the result
            child_to_expand.backpropagate(1 if result_colour == self.colour else -1)

        best_child = root.best_child(c=0)
        return Move(best_child.move[0], best_child.move[1])
