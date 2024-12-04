from src.Colour import Colour
from agents.Group23.AlphaZeroAgent import AlphaZeroAgent
from src.Colour import Colour
from src.Game import Game
from src.Player import Player
from agents.Group23.Alpha_Zero_NN import Alpha_Zero_NN

class alpha_zero_self_play_loop:
    _board_size: int = 11
    _Student_Network = None
    _Teacher_Network = None
    _max_games_per_simulation = 1
    _simulation_iterations = 100
    _game_log_location = "alpha_zero_self_play.log"

    def __init__(self):
        self._Student_Network = Alpha_Zero_NN(board_size=self._board_size)
        self._Teacher_Network = Alpha_Zero_NN(board_size=self._board_size)

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

        winner_colour = current_game.board.get_winner()

        return winner_colour


    def _run(self):
        for sim_iter in range(self._simulation_iterations):
            print(f"Simulation iteration {sim_iter+1} of {self._simulation_iterations}")

            win_count = 0

            for i in range(self._max_games_per_simulation):
                print(f"Game {i+1} of {self._max_games_per_simulation}")
                winner_colour = self._simulate_game()

                print(f"Student network won: {winner_colour == Colour.RED}")
                if winner_colour == Colour.RED:
                    win_count += 1

                self._Student_Network._commit_experience_from_buffer(winner_colour=Colour.RED)
                self._Teacher_Network._commit_experience_from_buffer(winner_colour=Colour.BLUE)

            # check majority win rate and swap networks if necessary
            if win_count > self._max_games_per_simulation / 2:
                self._swap_student_teacher_networks()

            # train student network
            self._Student_Network._train()

    def _swap_student_teacher_networks(self):
        # swap student and teacher networks
        print("Swapping student and teacher networks")
        temp = self._Student_Network
        self._Student_Network = self._Teacher_Network
        self._Teacher_Network = temp