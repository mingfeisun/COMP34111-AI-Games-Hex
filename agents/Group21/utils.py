from src.Colour import Colour
import numpy as np

def encode_board(board, current_player):
        size = board.size
        player_plane = []
        opponent_plane = []
        # Red is player1, Blue is player2
        player_to_move_plane = [1 if current_player == Colour.RED else 0] * (size * size)

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
        
        # Return a 3D numpy array with shape (3, size, size)
        return [np.array(player_plane), np.array(opponent_plane), np.array(player_to_move_plane)]