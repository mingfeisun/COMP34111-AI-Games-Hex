import os
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Input, Model
import tensorflow as tf
import pandas as pd
from src.Colour import Colour
import ast

class Alpha_Zero_NN:
    _board_size = 11 # default board size
    _model = None # current model
    _experience_data_buffer = []

    def load_experience_from_file(self, path:str):
        if os.path.exists("experience.txt"):
            return pd.read_csv("experience.txt")
        else:
            return pd.DataFrame(columns=['board_state', 'mcts_prob', 'z_value'])

    def __init__(self, board_size:int):
        """Initializes the AlphaZero neural network model

        Args:
            board_size (int): The size of the board
        """
        self._board_size = board_size
        self._model = self._create_model()

        self._train()

    def _create_policy_head(self, input_layer:tf.Tensor) -> tf.Tensor:
        """Creates the policy head of the model

        Args:
            input_layer (tf.Tensor): The input layer of the model

        Returns:
            tf.Tensor: The policy head of the model
        """
        x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = Flatten()(x)
        policy_output = Dense(self._board_size * self._board_size, activation='softmax', name='policy_head')(x)
        return policy_output
    
    def _create_value_head(self, input_layer:tf.Tensor) -> tf.Tensor:
        """Creates the value head of the model

        Args:
            input_layer (tf.Tensor): The input layer of the model

        Returns:
            tf.Tensor: The value head of the model
        """

        x = Conv2D(1, kernel_size=1, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        value_output = Dense(1, activation='tanh', name='value_head')(x)
        return value_output

    def _create_model(self):
        """Creates the model with the policy and value heads

        Returns:
            tf.keras.Model: The model with the policy and value heads
        """
        input_layer = Input(shape=(self._board_size, self._board_size, 1))  # Input for the board state

        # Policy and Value heads
        policy_head = self._create_policy_head(input_layer)
        value_head = self._create_value_head(input_layer)

        # Combine into a single model
        model = Model(inputs=input_layer, outputs=[value_head, policy_head])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss={
                'value_head': 'mean_squared_error',
                'policy_head': 'categorical_crossentropy',
            },
            metrics={
                'value_head': 'mean_absolute_error',
                'policy_head': 'accuracy',
            }
        )
        
        model.summary()

        return model
    
    def _add_experience_to_buffer(self, board_state:list[int], mcts_prob:list[float], player_colour:Colour):
        """Add single experience to the buffer

        Args:
            board_state (list[int]): current_board_state
            mcts_prob (list[float]): mcts_probabilities
            player_colour (Colour): player colour
        """
        self._experience_data_buffer.append((board_state, mcts_prob, player_colour))

    def save_experience_to_file(self, path:str, _game_experience:pd.DataFrame):
        """Save batch experience to file

        Args:
            path (str): path to save the file
        """
        _game_experience.to_csv(path, index=False)

    def _commit_experience_from_buffer(self, winner_colour:float):
        """Commit experience from buffer to game experience

        Args:
            winner_colour (float): game winner to update z values
        """

        _game_experience = self.load_experience_from_file("experience.txt")

        for board_state, mcts_prob, player_colour in self._experience_data_buffer:

            if player_colour == winner_colour:
                new_row = [board_state, mcts_prob, 1]
                _game_experience = pd.concat([_game_experience, pd.DataFrame([new_row], columns=_game_experience.columns)])
            else:
                new_row = [board_state, mcts_prob, -1]
                _game_experience = pd.concat([_game_experience, pd.DataFrame([new_row], columns=_game_experience.columns)])

        self.save_experience_to_file("experience.txt", _game_experience)

        self._experience_data_buffer = [] # clear buffer


    def get_train_val_data(self, validation_split=0.2):
        _game_experience = self.load_experience_from_file("experience.txt")

        if len(_game_experience) == 0:
            return [], [], [], [], [], []
        
        training_data = _game_experience.sample(frac=1-validation_split)
        validation_data = _game_experience.drop(training_data.index)

        # Convert board_state from string to array
        train_board_states = training_data['board_state'].apply(ast.literal_eval).tolist()
        train_z_values = training_data['z_value']
        train_mcts_probs = training_data['mcts_prob'].apply(ast.literal_eval).tolist()

        val_board_states = validation_data['board_state'].apply(ast.literal_eval).tolist()
        val_z_values = validation_data['z_value']
        val_mcts_probs = validation_data['mcts_prob'].apply(ast.literal_eval).tolist()

        print(f"Training data size: {len(train_board_states)}")
        print(f"Validation data size: {len(val_board_states)}")

        train_board_states = np.array(train_board_states)
        train_z_values = np.array(train_z_values)
        train_mcts_probs = np.array(train_mcts_probs)
        train_mcts_probs_flattened = [sample.ravel().tolist() for sample in train_mcts_probs]
        train_mcts_probs_flattened = np.array(train_mcts_probs_flattened)

        val_board_states = np.array(val_board_states)
        val_z_values = np.array(val_z_values)
        val_mcts_probs = np.array(val_mcts_probs)
        val_mcts_probs_flattened = [sample.ravel().tolist() for sample in val_mcts_probs]
        val_mcts_probs_flattened = np.array(val_mcts_probs_flattened)

        return train_board_states, train_z_values, train_mcts_probs_flattened, val_board_states, val_z_values, val_mcts_probs_flattened
    
    def _train(self, validation_split = 0.2):
        """Trains the model with the given data
        """
        train_board_states, train_z_values, train_mcts_probs, val_board_states, val_z_values, val_mcts_probs = self.get_train_val_data(validation_split)

        if len(train_board_states) == 0:
            print("No training data available")
            return

        self._model.fit(
            x=train_board_states,
            y={
                'value_head': train_z_values,
                'policy_head': train_mcts_probs
            },
            batch_size=64,
            epochs=10,
            validation_data=(
                val_board_states,
                {
                    'value_head': val_z_values,
                    'policy_head': val_mcts_probs
                }
            )
        )
        

    def get_policy_value(self, board_state:list[list[int]]) -> list[list[float]]:
        """Gets the policy and value for the given board state

        Args:
            board_state (list[list[int]]): experienced board states

        Returns:
            list[list[float]]: The policy for the given board state
        """
        if self._model is None:
            random_policy = []
            for _ in range(self._board_size):
                random_policy.append([1/self._board_size]*self._board_size)
            return random_policy
        
        board_state = np.array(board_state)
        board_state_reshaped = board_state.reshape(1,  board_state.shape[0], board_state.shape[1], 1)
        _, policy = self._model.predict(board_state_reshaped, verbose = 0)

        policy_reshaped = policy[0].reshape(11, 11)

        return policy_reshaped
    
    def get_predicted_value(self, board_state:list[list[int]]) -> float:
        """Get the predicted value for the given board state

        Args:
            board_state (list[list[int]]): experienced board states

        Returns:
            float: The predicted value for the given board state
        """
        if self._model is None:
            return 0.5
        
        # board_state = tf.convert_to_tensor(board_state, dtype=tf.float32)
        # board_state = tf.expand_dims(board_state, axis=0)
        board_state = np.array(board_state)
        board_state_reshaped = board_state.reshape(1,  board_state.shape[0], board_state.shape[1], 1)
        value, _ = self._model.predict(board_state_reshaped, verbose = 0)

        return value[0]
        
    def save_model(self, path:str):
        """Saves the model to the given path

        Args:
            path (str): The path to save the model
        """
        self._model.save(path)

    def load_model(self, path:str):
        """Loads the model from the given path

        Args:
            path (str): The path to load the model
        """
        self._model = tf.keras.models.load_model(path)