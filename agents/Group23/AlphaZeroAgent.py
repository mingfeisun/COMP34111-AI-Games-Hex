import logging

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.Group23.Alpha_Zero_NN import Alpha_Zero_NN

from agents.TestAgents.utils import make_valid_move
from agents.Group23.treenode import TreeNode
from agents.Group23.mcts import MCTS

class AlphaZeroAgent(AgentBase):
    logger = logging.getLogger(__name__)

    _board_size: int = 11
    _trained_policy_value_network = None # store a trained policy and value network
    _agent_in_training = False # flag to indicate if agent is in training mode
    _tree = None
    _turn_length = 1

    def __init__(self, colour: Colour, custom_trained_network: Alpha_Zero_NN = None, turn_length_s: int = 1):
        super().__init__(colour)
        if custom_trained_network is not None:
            self._trained_policy_value_network = custom_trained_network
            self._agent_in_training = True
        self._turn_length = turn_length_s # max length of a turn in seconds
        self._tree = None # MCTS tree

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
        board_state = self.get_board_vector(board_state)
        self._trained_policy_value_network._add_experience_to_buffer(board_state, mcts_probs, self.colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:

        if self._tree is None and opp_move is not None:
            logging.info("Initialising game tree...")
            self._tree = TreeNode(player=self.colour.opposite())
            self._tree = self.update_tree(self._tree, opp_move)

            x, y = opp_move.x, opp_move.y
            board.set_tile_colour(x, y, self.colour.opposite())
        elif self._tree is None:
            logging.info("Initialising game tree...")
            self._tree = TreeNode(player=self.colour)

        mcts = MCTS(board, self.colour, turn_length_s=self._turn_length)
        self._tree = mcts.run(self._tree)

        x, y = self._tree.move.x, self.tree.move.y
        board.set_tile_colour(x, y, self.colour)

        # self._record_experience(board, self._tree.mcts_probs)

        return self._tree.move
    
    def update_tree(self, tree: TreeNode, move: Move):
        """Updates the tree with the opponent's move."""
        for child in tree.children:
            if child.move == move:
                return child
        
        # If the move is not in the tree, create a new node
        return TreeNode(parent=tree, move=move)
