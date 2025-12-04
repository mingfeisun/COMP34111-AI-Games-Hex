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
        self.exploration_weight = exploration_weight

    def run(self, time_limit: float = 0.5, iterations: int = 1000) -> Move:
        assert self.root is not None, "Call update(board, opp_move) before run() to set root."

        end_time = time.time() + time_limit
        iters_left = iterations

        while iters_left > 0 and time.time() < end_time:
            leaf = self._select()
            child = self._expand(leaf) if not leaf.is_terminal else leaf
            reward = self._simulate(child)
            self._backpropagate(child, reward)
            iters_left -= 1

        if not self.root.children:
            fallback_move = choice(self.root.unexplored_moves)
            return fallback_move

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
        log_N_vertex = math.log(node.N + 1e-9)

        def uct(n: MCTSNode) -> float:
            """Returns the upper confidence bound for trees"""
            return (n.Q / (n.N + 1e-9)) + self.exploration_weight * math.sqrt(log_N_vertex / (n.N + 1e-9))

        return max(node.children.values(), key=uct)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        scored = [
            (self._move_heuristic(node.board, mv), mv)
            for mv in node.unexplored_moves
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_move = scored[0]
        return node.make_move(best_move)
    
    def _move_heuristic(self, board: Board, move) -> float:
        if hasattr(move, "x") and hasattr(move, "y"):
            x, y = move.x, move.y
        else:
            x, y = move

        n = board.size

        # Compute distance to the target winning side
        if self.colour == Colour.RED:
            dist = min(y, n-1-y)
        else:
            dist = min(x, n-1-x)

        # Computes center preference (the more center the better)
        cx = abs(x - n/2)
        cy = abs(y - n/2)
        center_score = - (cx + cy)

        return -dist + 0.3 * center_score

    def _simulate(self, node: MCTSNode) -> float:
        board = copy.deepcopy(node.board)
        current_colour = node.colour

        # Checking that the board hasn't ended due to the previous colour's play
        while not board.has_ended(Colour.opposite(current_colour)):
            moves = self._biased_simulation_moves(board, current_colour)
            move = choice(moves)
            board.set_tile_colour(move[0], move[1], current_colour)
            current_colour = Colour.opposite(current_colour)

        return 1 if board.get_winner() == self.root.colour else -1
    
    def _biased_simulation_moves(self, board: Board, colour: Colour):
        # List of legal moves
        empty = [
            (i, j)
            for i in range(board.size)
            for j in range(board.size)
            if not board.tiles[i][j].colour
        ]

        # Prefer moves adjacent to existing own color
        good = []
        for x, y in empty:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < board.size and 0 <= ny < board.size:
                    if board.tiles[nx][ny].colour == colour:
                        good.append((x, y))
                        break

        if good:
            return good

        # If no adjacent moves exist, use heuristic scoring
        scored = [(self._move_heuristic(board, (x, y)), (x, y)) for (x, y) in empty]
        scored.sort(reverse=True)

        # Keep only the best move at the top
        top = [mv for _, mv in scored[:max(4, len(scored)//5)]]
        return top
    
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
