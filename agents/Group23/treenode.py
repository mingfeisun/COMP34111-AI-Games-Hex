import math

class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, move=None, parent=None):
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_param=math.sqrt(2)):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) + exploration_param * math.sqrt(math.log(self.visits) / child.visits)
        )

    def add_child(self, move):
        """Adds a child node for a move."""
        child_node = TreeNode(move, parent=self)
        self.children.append(child_node)
        return child_node
    
######################################################################################