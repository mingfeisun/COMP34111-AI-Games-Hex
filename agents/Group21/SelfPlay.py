import numpy as np
from src.Colour import Colour 
from agents.Group21.MCTSAlphaZeroAgent import MCTSAlphaZeroAgent
from src.Move import Move
from src.Player import Player

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
            legal_moves, pi = current_agent.mcts.run(game, simulations=self.simulations)

            # store training sample but without z yet
            states.append(self.encode_board(game.board, current_player_colour))
            policies.append(pi)
            players_to_move.append(current_player_colour)

            # Sample a move from the improved policy pi
            move_index = np.random.choice(len(legal_moves), p=pi)
            move = legal_moves[move_index]

            game.current_player = current_player_colour
            game._make_move(Move(move[0], move[1]))
            current_player_colour = Colour.opposite(current_player_colour)

        # game finished, determine winner
        winner = game.board.get_winner()

        # Assign z for each move from the perspective of the player who made it
        for s, p, player in zip(states, policies, players_to_move):
            z = 1 if player == winner else -1
            examples.append((s, p, z))

        return examples
    
    def encode_board(self, board, current_player):
        size = board.size
        player_plane = []
        opponent_plane = []

        for i in range(size):
            for j in range(size):
                if board.tiles[i][j].colour == current_player:
                    player_plane.append(1)
                    opponent_plane.append(0)
                elif board.tiles[i][j].colour == Colour.opposite(current_player):
                    player_plane.append(0)
                    opponent_plane.append(1)
                else:
                    player_plane.append(0)
                    opponent_plane.append(0)
        
        return player_plane + opponent_plane