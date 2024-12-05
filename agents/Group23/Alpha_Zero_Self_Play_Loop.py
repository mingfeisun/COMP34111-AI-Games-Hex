from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.Group23.AlphaZeroAgent import AlphaZeroAgent
from src.Colour import Colour
from src.Game import Game
from src.Player import Player

class alpha_zero_self_play_loop:
    _board_size: int = 11
    _board_history = []
    _Student_Network = None
    _Teacher_Network = None
    _max_games = 10
    _game_log_location = "alpha_zero_self_play.log"

    def __init__(self):
        self._Student_Network = ...
        self._Teacher_Network = ...

    def set_up_game(self):
        g = Game(
            player1=Player(
                name="student player",
                agent=AlphaZeroAgent(Colour.RED, custom_trained_network = self._Student_Network),
            ),
            player2=Player(
                name="teacher player",
                agent=AlphaZeroAgent(Colour.BLUE, custom_trained_network = self._Teacher_Network),
            ),
            board_size=self._board_size,
            logDest=self._game_log_location,
            verbose=True,
            silent=True
        )

        return g

    def _simulate_game(self):
        # simulate game between two agents
        current_game = self.set_up_game()
        current_game.run()

        board_state = current_game.board.tiles
        winner_colour = current_game.board.get_winner()

        if winner_colour == Colour.RED:
            return {
                "z": 1,
                "board_state": board_state
            }
        
        if winner_colour == Colour.BLUE:
            return  {
                "z": -1,
                "board_state": board_state
            }
        
        Exception("Invalid winner colour returned from self play game")


    def _run(self):
        for i in range(self._max_games):
            print(f"Game {i+1} of {self._max_games}")
            game_result = self._simulate_game()
            self._board_history.append(game_result)

            print(f"Student network won: {game_result['z'] == 1}")
            if game_result["z"] == -1:
                self._swap_student_teacher_networks()


    def _swap_student_teacher_networks(self):
        # swap student and teacher networks
        print("Swapping student and teacher networks")
        temp = self._Student_Network
        self._Student_Network = self._Teacher_Network
        self._Teacher_Network = temp