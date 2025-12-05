import copy
import math
import time
from random import choice

from agents.Group21.MCTSNode import MCTSNode
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTS:
    def __init__(self, colour: Colour, exploration_weight: float = 1):
        self.colour = colour
        self.root: MCTSNode | None = None
        self.current_turn = 0
        self.exploration_weight = exploration_weight

    # TODO: Time limit or iterations?
    # TODO: From first run we need at minimum 11^2 - 1 iterations before all moves can even be chosen
    # Perhaps we can have iterations / time_limit decay as rounds go on?
    # TODO: How do we pick child? Highest N or Q/N?
    def run(self, time_limit: float = 0.5, iterations: int = 20) -> Move:
        # end_time = time.time() + time_limit
        # while time.time() < end_time:
        for _ in range(iterations):
            node = self._select()

            # Skip expansion if the node is already terminal
            if not node.is_terminal:
                node_to_expand = self._expand(node)
                reward = self._simulate(node_to_expand)
                self._backpropagate(node_to_expand, reward)
            else:
                reward = self._simulate(node)
                self._backpropagate(node, reward)

        # Picking the child with the highest visit count
        best_move, best_child = max(self.root.children.items(), key=lambda child: child[1].N)
        print(f'Best move was {best_move} with Q={best_child.Q}, N={best_child.N}')
        return best_move

    # TODO: Should we prune the old states? Do we need to keep track?
    def update(self, board: Board, opp_move: Move | None) -> None:
        """Given a move, find the corresponding child of the root and set that as the new root"""

        # Initial set up
        if self.root is None:
            self.root = MCTSNode(self.colour, board)
            return

        # Check if we have a node for the opponent's move, in which case we can reuse
        if opp_move in self.root.children:
            self.root = self.root.children[opp_move]
            self.root.parent = None
            return

        # Otherwise, create a new node
        else:
            self.root = MCTSNode(self.colour, board)

    def _select(self) -> MCTSNode:
        """Find an unexplored descendent of the root node"""
        node = self.root
        while not node.is_terminal and node.is_fully_explored:
            node = self._uct_select(node)
        return node

    # TODO: Division by 0 for N = 0?
    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """Select a child of node, balancing exploration & exploitation"""
        log_N_vertex = math.log(node.N)

        def uct(n: MCTSNode) -> float:
            """Returns the upper confidence bound for trees"""
            return (n.Q / n.N) + self.exploration_weight * math.sqrt(log_N_vertex / n.N)

        return max(node.children.values(), key=uct)

    # TODO: A heuristic to pick more promising areas first if MCTS takes too long
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Returns a randomly chosen node from the available moves"""
        move_to_expand = choice(node.unexplored_moves(self.current_turn == 2))
        return node.make_move(move_to_expand)

    # TODO: 0/1 for rewards or -1/+1?
    # TODO: Possibly use a heuristic to allow for a reward even if node is non-terminal
    # TODO: Heuristic for move instead of random?
    # TODO: Parallelize
    def _simulate(self, node: MCTSNode) -> float:
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Checking that the board hasn't ended due to the previous colour's play
        while not board.has_ended(Colour.opposite(current_colour)):
            possible_moves = [
                (i, j)
                for i in range(board.size)
                for j in range(board.size)
                if not board.tiles[i][j].colour
            ]

            x, y = choice(possible_moves)
            board.set_tile_colour(x, y, current_colour)
            current_colour = Colour.opposite(current_colour)

        # Get reward
        return 1 if board.get_winner() == self.root.colour else -1

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float):
        """Backpropagates rewards and visits until the root node is reached"""
        current_node = node
        current_reward = reward
        while current_node is not None:
            current_node.Q += current_reward
            current_node.N += 1

            current_node = current_node.parent
            current_reward = -current_reward # Flip reward as 0-sum
