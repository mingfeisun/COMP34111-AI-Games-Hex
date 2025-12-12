import numpy as np
from src.Board import Board

class RandomNN:
    def predict(self, state):
        '''Given a state tensor, return a random policy and value.
        
        Args:
            state: shape (3, size, size) numpy array representing the board state.
        '''
        
        size = 11

        probs = np.ones(size * size) / (size * size)
        v = 0  

        return probs, v