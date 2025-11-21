from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()
    
    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None
    
    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True
    
    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss"
        return 0
    
    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789
    
    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True