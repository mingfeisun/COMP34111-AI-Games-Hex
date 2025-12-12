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

    # Convert to numpy arrays and reshape to (11, 11)
    player_plane = np.array(player_plane).reshape(size, size)
    opponent_plane = np.array(opponent_plane).reshape(size, size)
    player_to_move_plane = np.array(player_to_move_plane).reshape(size, size)

    # Stack planes to get shape (3, 11, 11)
    return np.stack([player_plane, opponent_plane, player_to_move_plane], axis=0)