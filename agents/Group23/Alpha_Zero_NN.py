import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import Input, Model
import tensorflow as tf 

class Alpha_Zero_NN:
    _board_size = 11 # default board size
    _model = None # current model
    _game_experience = [] # list of game experience

    def __init__(self, board_size:int):
        """Initializes the AlphaZero neural network model

        Args:
            board_size (int): The size of the board
        """
        self._board_size = board_size
        self._model = self._create_model()

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
    
    def add_game_experience(self, board_states:list[list[list[int]]], z_values:list[float], mcts_probs:list[list[float]]):
        """Adds the game experience to the list of game experience

        Args:
            board_states (list[list[list[int]]]): experienced board states
            z_values (list[float]): associated z values with the board states
            mcts_probs (list[list[float]]): associated mcts probabilities with the board states
        """
        self._game_experience.append((board_states, z_values, mcts_probs))

    def get_train_val_data(self, validation_split=0.2):
        board_states = [
            board_state for game in self._game_experience for board_state in game[0]
        ]

        z_values = [
            z_value for game in self._game_experience for z_value in game[1]
        ]

        mcts_probs = [
            mcts_prob for game in self._game_experience for mcts_prob in game[2]
        ]

        zip_data = list(zip(board_states, z_values, mcts_probs))
        np.random.shuffle(zip_data)

        split_index = int(len(zip_data) * (1 - validation_split))
        training_data = zip_data[:split_index]
        validation_data = zip_data[split_index:]

        train_board_states, train_z_values, train_mcts_probs = zip(*training_data)
        train_board_states = np.array(train_board_states)
        train_z_values = np.array(train_z_values)
        train_mcts_probs = np.array(train_mcts_probs)

        val_board_states, val_z_values, val_mcts_probs = zip(*validation_data)
        val_board_states = np.array(val_board_states)
        val_z_values = np.array(val_z_values)
        val_mcts_probs = np.array(val_mcts_probs)


        return train_board_states, train_z_values, train_mcts_probs, val_board_states, val_z_values, val_mcts_probs
    
    def train(self, validation_split = 0.2):
        """Trains the model with the given data
        """
        train_board_states, train_z_values, train_mcts_probs, val_board_states, val_z_values, val_mcts_probs = self.get_train_val_data(validation_split)

        self.model.fit(
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
        if self._policy_head is None or self._value_head is None:
            random_policy = []
            for i in range(self._board_size):
                random_policy.append([1/self._board_size]*self._board_size)
            return random_policy
        
        board_state = tf.convert_to_tensor(board_state, dtype=tf.float32)
        board_state = tf.expand_dims(board_state, axis=0)
        policy, _ = self._model.predict(board_state)

        return policy[0]
    
    def get_predicted_value(self, board_state:list[list[int]]) -> float:
        """Get the predicted value for the given board state

        Args:
            board_state (list[list[int]]): experienced board states

        Returns:
            float: The predicted value for the given board state
        """
        if self._policy_head is None or self._value_head is None:
            return 0.5
        
        board_state = tf.convert_to_tensor(board_state, dtype=tf.float32)
        board_state = tf.expand_dims(board_state, axis=0)
        _, value = self._model.predict(board_state)

        return value[0][0]
        
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


if __name__ == "__main__":     
    _ = Alpha_Zero_NN(11) # test model compiles