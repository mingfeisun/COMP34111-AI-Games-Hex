import logging
import math

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group23.Alpha_Zero_NN import Alpha_Zero_NN
from agents.Group23.alpha_zero_mcts import MCTS

class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, move=None, parent=None, player=None):
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node
        self.q_rave = 0  # times this move has been critical in a rollout
        self.n_rave = 0  # times this move has been played in a rollout
        self.player = player
        self.rave_const = 300
        
        if parent is not None:
            self.player = parent.player.opposite()

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, explore=math.sqrt(2)):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: self.__value__(child, explore)
        )

    def add_child(self, move):
        """Adds a child node for a move."""
        child_node = TreeNode(move, parent=self)
        self.children.append(child_node)
        return child_node
    
    def __value__(self, child, explore=math.sqrt(2), heuristic='uct'):
        uct = (child.wins / child.visits) + explore * math.sqrt(math.log(self.visits) / child.visits)

        if heuristic == 'uct':
            return uct
        elif heuristic == 'rave':
            alpha = max(0, (self.rave_const - self.N) / self.rave_const)
            amaf = self.q_rave / self.n_rave if self.n_rave != 0 else 0
            rave = (1 - alpha) * uct + alpha * amaf
            return rave

        return self.wins / self.visits if self.visits != 0 else 0
    

class AlphaZeroAgent(AgentBase):
    logger = logging.getLogger(__name__)

    _board_size: int = 11
    _trained_policy_value_network = None # store a trained policy and value network
    _agent_in_training = False # flag to indicate if agent is in training mode
    tree = None
    turn_length = 2

    def __init__(self, colour: Colour, custom_trained_network: Alpha_Zero_NN = None, turn_length_s: int = 1):
        super().__init__(colour)
        if custom_trained_network is not None:
            self._trained_policy_value_network = custom_trained_network
            self._agent_in_training = True
        self.turn_length = turn_length_s # max length of a turn in seconds
        self.tree = None # MCTS tree


    def get_board_vector(self, board: Board) -> list[list[int]]:
        """generate input vector for neural network
        based on current and recent board states

        Args:
            board (Board): current board state

        Returns:
            list[int]: input vector for neural network
        """
        
        # convert board state to input vector
        board_vector = []
        for i in range(self._board_size):
            new_line = []
            for j in range(self._board_size):
                tile = board.tiles[i][j].colour
                if tile == None:
                    new_line.append(0)
                elif tile == self.colour:
                    new_line.append(1)
                else:
                    new_line.append(-1)
            board_vector.append(new_line)

        return board_vector
    
    def _record_experience(self, board_state: Board, mcts_probs: list[list[float]]):
        """Record the experience for the given board state

        Args:
            board_state (list[list[int]]): experienced board states
            mcts_probs (list[list[float]]): mcts probabilities for the given board state
        """
        print(f"Recording experience ({self.colour})")
        board_vector_state = self.get_board_vector(board_state)
        self._trained_policy_value_network._add_experience_to_buffer(board_vector_state, mcts_probs, self.colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:

        if self.tree is None and opp_move is not None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour.opposite())
            self.tree = self.update_tree(self.tree, opp_move)

            x, y = opp_move.x, opp_move.y
            board.set_tile_colour(x, y, self.colour.opposite())
        elif self.tree is None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour)

        mcts = MCTS(board, self.colour, turn_length_s=self.turn_length, custom_trained_network=self._trained_policy_value_network)
        self.tree, visit_count_normalised_distribution = mcts.run(self.tree, )

        if self._agent_in_training:
            # add data to training buffer before modifying board
            self._record_experience(board, visit_count_normalised_distribution)


        x, y = self.tree.move.x, self.tree.move.y
        board.set_tile_colour(x, y, self.colour)


        return self.tree.move
    
    def update_tree(self, tree: TreeNode, move: Move):
        """Updates the tree with the opponent's move."""
        for child in tree.children:
            if child.move == move:
                return child
        
        # If the move is not in the tree, create a new node
        return TreeNode(parent=tree, move=move)
    