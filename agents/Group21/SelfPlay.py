import numpy as np
from src.Colour import Colour 
from agents.Group21.MCTSAlphaZeroAgent import MCTSAlphaZeroAgent
from src.Move import Move
from src.Player import Player
from agents.Group21.utils import encode_board

class SelfPlay:
    def __init__(self, neural_net, game_cls, simulations=800):
        self.nn = neural_net
        self.game_cls = game_cls
        self.simulations = simulations

    # Playing a single game
    def play_game(self):
        examples = []

        player1 = Player("Player1", MCTSAlphaZeroAgent(Colour.RED, self.nn, self.simulations))
        player2 = Player("Player2", MCTSAlphaZeroAgent(Colour.BLUE, self.nn, self.simulations))

        game = self.game_cls(player1, player2)

        # Lists to store training examples during the game
        states = []
        policies = []
        players_to_move = []

        current_player_colour= Colour.RED

        while not game.board.has_ended(current_player_colour):
            current_agent = game.players[current_player_colour].agent
            # get legal moves and improved policy from MCTS from current state 
            legal_moves, pi = current_agent.mcts.run(game, simulations=self.simulations)

            # store training sample but without z yet
            states.append(encode_board(game.board, current_player_colour))
            policies.append(pi)
            players_to_move.append(current_player_colour)

            # Sample a move from the improved policy pi
            move_index = np.argmax(pi)
            move = legal_moves[move_index]

            game.current_player = current_player_colour
            game._make_move(Move(move[0], move[1]))
            current_player_colour = Colour.opposite(current_player_colour)

        # game finished, determine winner
        winner = game.board.get_winner()

        # Assign z for each move from the perspective of the player who made the move, e.g. if Blue to move and Blue wins, z=1
        for s, p, player in zip(states, policies, players_to_move):
            z = 1 if player == winner else -1
            examples.append((s, p, z))

        return examples