from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class HeuristicMove:
    """Represents a player move in a turn of Hex.
    A swap move is when x=-1 and y=-1.
    Priority is used to partition moves for consideration by priority.
    Priority is assigned as follows:
    1. One-to-connect moves that connect the (1-edge-connected) chain to another edge
       + One-possible-connect moves that can connect the (1-edge-connected) chain to another chain which is connected to the other edge 

    2. One-to-connect moves that connect the chain to a new edge
    3. One-to-connect moves that connect the chain to another chain which is connected to an edge

    4. One-possible-connect moves that can connect the chain to to a new edge
    5. One-possible-connect moves that can connect the chain to another chain which is connected to an edge

    6. One-to-connect moves that connect the chain to another chain
       + One-possible-connect moves that can connect the chain to another chain
       + Random moves

    Inspired by heuristic priorities in outlined in "Apply Heuristic Search to Discover a New Winning Solution in Hex":
    https://doi.org/10.1109/FSKD.2007.185

    TODO: 
    Consider priority level 6. 
     - Should the one-to-connect and one-possible-connect moves not linking to an edge be considered?
     - How do we determine if it is best to:
        - Extend an existing chain
        - Connect two chains (which are not connected to an edge)
        - Create a new chain
     - Currently only using random moves (choice from all available moves) for priority level 6.
    """

    _x: int = -1
    _y: int = -1
    _priority: int = -1

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y
    
    @property
    def priority(self) -> int:
        return self._priority
