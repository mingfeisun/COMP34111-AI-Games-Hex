import pytest

from agents.Group25.tests.conftest import blue_win_board
from src.Board import Board
from src.Colour import Colour
from agents.Group25.MCTSAgent import MCTSNode, get_valid_moves, get_opponent_colour, MCTSAgent


class TestHelperMethods:
    """
    Test helper methods
    """

    def test_get_valid_moves(self, blue_win_board: Board):
        """
        Test the get_valid_moves method
        """
        valid_moves = get_valid_moves(blue_win_board)
        assert len(valid_moves) == 27
        assert set(valid_moves) == {(3, 4),
                                     (3, 7),
                                     (4, 9),
                                     (5, 5),
                                     (5, 7),
                                     (5, 8),
                                     (5, 9),
                                     (5, 10),
                                     (6, 0),
                                     (6, 2),
                                     (6, 9),
                                     (7, 2),
                                     (7, 4),
                                     (7, 7),
                                     (7, 8),
                                     (7, 10),
                                     (8, 0),
                                     (8, 1),
                                     (8, 6),
                                     (8, 10),
                                     (9, 6),
                                     (9, 10),
                                     (10, 0),
                                     (10, 5),
                                     (10, 6),
                                     (10, 8),
                                     (10, 10)}

    def test_get_opponent_colour(self):
        """
        Test the get_opponent_colour method
        """
        assert get_opponent_colour(Colour.RED) == Colour.BLUE
        assert get_opponent_colour(Colour.BLUE) == Colour.RED
        with pytest.raises(ValueError):
            get_opponent_colour(3)

class TestMCTSNode:
    """
    Test the MCTSNode class
    """
    def test_generate_all_children(self):
        """
        Test the generate_all_children_nodes method
        """
        state = Board()
        node = MCTSNode(state=state, move=None)
        node.generate_all_children_nodes(Colour.RED)
        assert len(node.children) == len(node.valid_moves)

    def test_backpropagate(self):
        """
        Test the backpropagate method
        """
        state = Board()
        node = MCTSNode(state=state, move=None)
        node.backpropagate(1)
        assert node.visits == 1
        assert node.payoff_sum == 1

        node.backpropagate(-1)
        assert node.visits == 2
        assert node.payoff_sum == 0

    def test_simulate_from_node(self):
        """
        Test the simulate_from_node method
        """
        state = Board()
        node = MCTSNode(state=state, move=None)
        result = node.simulate_from_node(Colour.RED)
        assert result == Colour.RED or result == Colour.BLUE

    def test_get_best_child_unvisited_nodes(self):
        """
        Test the get_best_child method
        """
        state = Board()
        node = MCTSNode(state=state, move=None)
        node.generate_all_children_nodes(Colour.RED)
        best_child = node.best_child(1.41)
        assert best_child.visits == 0

    def test_get_best_child_visited_nodes(self):
        """
        Test the get_best_child method
        """
        state = Board()
        node = MCTSNode(state=state, move=None)
        node.generate_all_children_nodes(Colour.RED)
        for child in node.children:
            child.visits = 1
        best_child = node.best_child(1.41)
        assert best_child.visits == 1


'''
These tests are currently commented out as moves are too random to test reliably

# Moves need to be made within 5 seconds
# Will change when the agent is changed to calculate how much time it has to make a move
@pytest.mark.timeout(5)
class TestMCTSAgent:
    """
    Test the MCTSAgent class
    """
    def test_make_move_swap(self):
        """
        Test the make_move swaps on the second turn

        Will have to be changed when the MCTSAgent is updated to have logic for the swap
        """
        state = Board()
        agent = MCTSAgent(Colour.RED)
        move = agent.make_move(turn=2, board=state, opp_move=None)
        assert move.x == -1
        assert move.y == -1

    def test_make_move(self):
        """
        Test the make_move method
        """
        state = Board()
        agent = MCTSAgent(Colour.RED)
        move = agent.make_move(turn=3, board=state, opp_move=None)
        possible_moves = get_valid_moves(state)
        assert (move.x, move.y) in possible_moves

    def test_make_move_one_away(self, one_away_board):
        """
        Test the make_move method when the agent is one move away from winning
        """
        agent = MCTSAgent(Colour.BLUE)
        move = agent.make_move(turn=4, board=one_away_board, opp_move=None)
        assert move.x == 6
        assert move.y == 9
'''
