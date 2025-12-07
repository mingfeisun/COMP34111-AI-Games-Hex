import math
import numpy as np
import copy
from src.Colour import Colour 

class AlphaZeroMCTS:
    def __init__(self, neural_net, cpuct=1.0):
        self.neural_net = neural_net
        self.cpuct = cpuct
        self.Q = {}   # Q(s,a)
        self.Ns = {}  # total visits N(s)
        self.Nsa = {} # visits N(s,a)
        self.Ps = {}  # policy P(s)
        self.Es = {}  # game termination
        self.Vs = {}  # legal moves

    def run(self, game, simulations=800):
        '''Runs X simulations starting from the current game state'''
        board = game.board
        player = game.current_player

        for _ in range(simulations):
            self.search(board, player)

        s = self._board_to_string(game.board)

        # After simulations, build improved policy pi (π)
        counts = [self.Nsa.get((s, a), 0) for a in self.Vs[s]]
        pi = np.array(counts) / sum(counts) if sum(counts) > 0 else np.ones(len(counts)) / len(counts)
        return self.Vs[s], pi

    # Do one simulation given a state in game
    def search(self, board, player):
        s = self._board_to_string(board)

        # Check if its game termination
        if s not in self.Es:
            self.Es[s] = board.get_winner()  # +1 red, -1 blue, None continues

        # If game is over, return this outcome (+1, -1) immediately
        if self.Es[s] is not None:
            return 1 if self.Es[s] == player else -1

        # First time visiting this state
        if s not in self.Ps:

            # Get all legal moves
            legal_moves = [
                (i, j)
                for i in range(board.size)
                for j in range(board.size)
                if not board.tiles[i][j].colour
            ]
            self.Vs[s] = legal_moves

            # Evaluate state using NN
            policy, v = self.neural_net.predict(board)
            # Creates a probability distribution only over legal moves
            size = board.size
            p = np.array([policy[i * size + j] for (i, j) in legal_moves])
            p = p / np.sum(p)

            # Store prior (policy)
            self.Ps[s] = {a: p[i] for i,a in enumerate(legal_moves)}
            self.Ns[s] = 0

            # Return NN value v
            return v

        if not self.Vs[s]:  # no legal moves
            return 0

        # Otherwise select using UCT
        best = -float('inf')
        best_move = None

        for a in self.Vs[s]:
            if (s,a) in self.Q:
                u = self.Q[(s,a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if u > best:
                best = u

                # Select the max U
                best_move = a

        # Recurse to next state, select children until terminal or unvisited node
        next_board = copy.deepcopy(board)
        next_board.set_tile_colour(best_move[0], best_move[1], player)
        next_player = Colour.opposite(player)

        v = self.search(next_board, next_player)

        # Backpropagate the returned value
        if (s,best_move) in self.Q:
            self.Q[(s,best_move)] = (self.Nsa[(s,best_move)] * self.Q[(s,best_move)] + v) / (self.Nsa[(s,best_move)] + 1)
            self.Nsa[(s,best_move)] += 1
        else:
            self.Q[(s,best_move)] = v
            self.Nsa[(s,best_move)] = 1

        self.Ns[s] += 1
        return v

    def _board_to_string(self, board):
        """Converts the board to a hashable string for dict keys"""
        return str([[tile.colour for tile in row] for row in board.tiles])