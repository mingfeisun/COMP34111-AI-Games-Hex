import logging

from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

from agents.Group23.treenode import TreeNode
from agents.Group23.mcts import MCTS

class MCTSAgent(AgentBase):
    """An agent that uses MCTS for Hex."""
    logger = logging.getLogger(__name__)

    def __init__(self, colour: Colour, turn_length_s: int = 1):
        super().__init__(colour)
        self.turn_length = turn_length_s # max length of a turn in seconds
        self.tree = None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Selects a move using MCTS."""
        if self.tree is None and opp_move is not None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour.opposite())
            self.tree = self.update_tree(self.tree, opp_move)

            x, y = opp_move.x, opp_move.y
            board.set_tile_colour(x, y, self.colour.opposite())
        elif self.tree is None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour)

        mcts = MCTS(board, self.colour, turn_length_s=self.turn_length)
        self.tree = mcts.run(self.tree)

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

