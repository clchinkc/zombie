

from collections import defaultdict

import numpy as np
from keras import layers, models, optimizers


# Define a game state representation
class GameState:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def get_legal_moves(self):
        legal_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def apply_move(self, move):
        i, j = move
        if self.board[i][j] == 0:
            self.board[i][j] = self.player
            self.player = -self.player
        else:
            raise ValueError("Invalid move")

    def game_over(self):
        return self.get_winner() is not None or len(self.get_legal_moves()) == 0

    def get_winner(self):
        for i in range(3):
            row_sum = np.sum(self.board[i, :])
            col_sum = np.sum(self.board[:, i])
            if row_sum == 3 or col_sum == 3:
                return 1
            elif row_sum == -3 or col_sum == -3:
                return -1

        diag_sum1 = np.sum(np.diag(self.board))
        diag_sum2 = np.sum(np.diag(np.fliplr(self.board)))
        if diag_sum1 == 3 or diag_sum2 == 3:
            return 1
        elif diag_sum1 == -3 or diag_sum2 == -3:
            return -1

        return None
    
    def copy(self):
        gs = GameState()
        gs.board = np.copy(self.board)
        gs.player = self.player
        return gs

# Define the neural network architecture
def create_neural_network(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)

    # first residual block
    x = layers.Conv2D(128, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    shortcut = x
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)

    # second residual block
    shortcut = x
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)

    # policy head
    policy_head = layers.Conv2D(2, 1, padding="same", activation="relu")(x)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(num_actions, activation="softmax")(policy_head)

    # value head
    value_head = layers.Conv2D(1, 1, padding="same", activation="relu")(x)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(128, activation="relu")(value_head)
    value_head = layers.Dense(1, activation="tanh")(value_head)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                loss=['categorical_crossentropy', 'mse'],
                loss_weights=[1.0, 1.0])

    return model


# Define the MCTS algorithm
class MCTS:
    def __init__(self, model, cpuct=1.0, num_simulations=1600):
        self.model = model
        self.cpuct = cpuct
        self.num_simulations = num_simulations

        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(int)
        self.Ns = defaultdict(int)
        self.Ps = {}

    def search(self, game_state):
        for _ in range(self.num_simulations):
            self._simulate(game_state)

    def select(self, game_state):
        s = self._state_key(game_state)
        legal_moves = game_state.get_legal_moves()

        best_move = None
        best_value = float('-inf')

        for move in legal_moves:
            sa = self._state_action_key(s, move)
            value = self.Qsa[sa] / (1 + self.Nsa[sa])
            if value > best_value:
                best_move = move
                best_value = value

        return best_move

    def _simulate(self, game_state):
        path = []
        current_state = game_state

        while not current_state.game_over():
            s = self._state_key(current_state)
            legal_moves = current_state.get_legal_moves()

            if s not in self.Ps:
                # Get neural network predictions
                input_data = self._state_to_input(current_state)
                probs, _ = self.model.predict(input_data, verbose=0)
                probs = np.squeeze(probs)
                legal_probs = {move: probs[self._move_to_index(move)] for move in legal_moves}
                self.Ps[s] = legal_probs

                total_N = sum(self.Nsa[self._state_action_key(s, move)] for move in legal_moves)
                ucb_values = [
                    (self.Qsa[self._state_action_key(s, move)] + self.cpuct * self.Ps[s][move] * np.sqrt(total_N) / (1 + self.Nsa[self._state_action_key(s, move)])) for move in legal_moves]
                move = legal_moves[np.argmax(ucb_values)]

            else:
                move = max(legal_moves, key=lambda m: self.Nsa[self._state_action_key(s, m)])

            path.append((s, move))
            current_state.apply_move(move)

        winner = current_state.get_winner()
        for s, move in path:
            sa = self._state_action_key(s, move)
            self.Nsa[sa] += 1
            self.Qsa[sa] += winner if s[0] == '1' else -winner
            self.Ns[s] += 1

    def _state_key(self, game_state):
        return str(game_state.player) + str(game_state.board.flatten())

    def _state_action_key(self, s, a):
        return s + str(a)

    def _state_to_input(self, game_state):
        # Convert game state to a suitable input for the neural network
        player_board = (game_state.board == game_state.player).astype(int)
        opponent_board = (game_state.board == -game_state.player).astype(int)
        empty_board = (game_state.board == 0).astype(int)
        input_data = np.stack([player_board, opponent_board, empty_board], axis=-1)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def _move_to_index(self, move):
        # Convert a move to an index for the probability distribution
        i, j = move
        return i * 3 + j

def play_self_play_game(model, mcts):
    game_data = []
    game_state = GameState()

    while not game_state.game_over():
        mcts.search(game_state.copy())  # Pass a copy of the game_state
        move = mcts.select(game_state)
        input_data = mcts._state_to_input(game_state)
        policy_probs, _ = model.predict(input_data, verbose=0)
        game_data.append((input_data, move, policy_probs))
        game_state.apply_move(move)

    winner = game_state.get_winner()
    for data_point in game_data:
        input_data, move, policy_probs = data_point
        target_value = winner if game_state.player == 1 else -(winner or 0)
        target_probs = np.zeros_like(policy_probs)
        target_probs[0, mcts._move_to_index(move)] = 1
        yield input_data, target_probs, target_value

def train_neural_network(model, data, num_epochs, batch_size):
    # Unzip the data
    input_data, target_probs, target_values = zip(*data)
    input_data = np.concatenate(input_data, axis=0)
    target_probs = np.concatenate(target_probs, axis=0)
    target_values = np.array(target_values)

    model.fit(input_data, [target_probs, target_values], batch_size=batch_size, epochs=num_epochs)


# Define the training loop
def train(model, mcts, num_iterations, num_games, num_epochs, batch_size):
    for iteration in range(num_iterations):
        data = []

        for game in range(num_games):
            # Play a game against itself using MCTS
            game_data = play_self_play_game(model, mcts)

            # Add game data to the dataset
            data.extend(game_data)

        # Train the neural network on the dataset
        train_neural_network(model, data, num_epochs, batch_size)

    return model

# Play against opponents
def play_game(mcts, model, game_state):
    while not game_state.game_over():
        print("Current game state:")
        print(game_state.board)

        if game_state.player == 1:
            # Use the trained neural network to play
            input_data = mcts._state_to_input(game_state)
            policy_probs, _ = model.predict(input_data, verbose=0)
            legal_moves = game_state.get_legal_moves()
            legal_probs = {move: policy_probs[0][mcts._move_to_index(move)] for move in legal_moves}
            move = max(legal_probs, key=lambda k: legal_probs[k])
            print(f"Model move: {move}")
        else:
            # Human or computer opponent's turn
            move = None
            legal_moves = game_state.get_legal_moves()
            while move not in legal_moves:
                move_input = input("Enter your move as row and column separated by space (e.g., '1 2'): ")
                move = tuple(map(int, move_input.strip().split()))
                if move not in legal_moves:
                    print("Invalid move, please try again.")
            print(f"Opponent move: {move}")

        game_state.apply_move(move)

    winner = game_state.get_winner()
    if winner == 1:
        print("The trained model wins!")
    elif winner == -1:
        print("The opponent wins!")
    else:
        print("It's a draw!")
        
    return game_state


if __name__ == "__main__":
    input_shape = (3, 3, 3)
    num_actions = 9

    # Create the neural network
    model = create_neural_network(input_shape, num_actions)

    # Create the MCTS algorithm
    mcts = MCTS(model)
    
    # Train the neural network
    model = train(model, mcts, num_iterations=10, num_games=10, num_epochs=10, batch_size=1024)
    
    # Play against opponents
    game_state = GameState()
    
    while not game_state.game_over():
        game_state = play_game(mcts, model, game_state)
        
    print("Winner: {}".format(game_state.get_winner()))
    
    # Save the trained model
    model.save("model.h5")
    
    
# Assume human is one side of chess and zombie is one side of chess. Use alphazero's monte carlo tree search.
# https://www.youtube.com/watch?v=wuSQpLinRB4
# https://tmoer.github.io/AlphaZero/
# https://github.com/timvvvht/AlphaZero-Connect4
# https://github.com/topics/alphazero
