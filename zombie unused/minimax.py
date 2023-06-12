

# minimax with alpha-beta pruning
# monte carlo
"""
Minimax and Monte Carlo are two different algorithms used in different contexts, but they are both commonly employed in game playing and decision-making problems.

1. Minimax Algorithm:
The minimax algorithm is a classic approach used in game theory and game playing. It is primarily employed in turn-based games with perfect information, such as chess or tic-tac-toe. The goal of the minimax algorithm is to find the optimal move for a player by considering all possible moves and their potential outcomes.

The minimax algorithm works by recursively evaluating the game state from the perspective of both players. It assumes that the opponent will make the best possible move, and the current player selects the move that minimizes the maximum potential loss (hence the name minimax). The algorithm constructs a game tree by exploring all possible moves and their subsequent moves, evaluating each leaf node using a heuristic function or an evaluation function.

While minimax guarantees an optimal solution in games with finite states, the main limitation is that it can be computationally expensive in games with large search spaces. To address this issue, optimizations like alpha-beta pruning are often employed to reduce the number of explored nodes.

2. Monte Carlo Algorithm:
Monte Carlo methods, on the other hand, are a broader class of algorithms that rely on random sampling to estimate outcomes or solve problems. Monte Carlo methods are often used in situations where deterministic or analytical approaches are not feasible or too complex.

In the context of game playing, Monte Carlo methods are commonly used in games with stochastic elements or incomplete information. Instead of considering all possible moves like minimax, Monte Carlo methods focus on estimating the value of moves through random sampling. The idea is to simulate many random games or scenarios and observe the outcomes to make informed decisions.

One popular Monte Carlo algorithm for game playing is Monte Carlo Tree Search (MCTS). MCTS combines elements of both minimax and Monte Carlo methods. It constructs a search tree iteratively, using a selection-expansion-simulation-backpropagation process. During the simulation phase, random playouts are performed to estimate the value of each move, and the results are backpropagated to update the search tree.

Monte Carlo methods can be computationally efficient and offer good performance in complex games with uncertain or partially observable states. However, their estimates may have a higher variance compared to the deterministic evaluations used in minimax.

In summary, minimax is a deterministic approach that explores all possible moves to find the optimal solution in games with perfect information, while Monte Carlo methods, such as MCTS, rely on random sampling to estimate move values and are often used in games with stochastic elements or incomplete information.
"""

import math

import numpy as np
import pygame
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, models, optimizers
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import math
import random

import numpy as np
import pygame
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, models, optimizers
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class Runner:
    def __init__(self, position):
        self.position = position

    def move(self, direction, is_valid_move):
        new_position = self._get_new_position(direction)
        if is_valid_move(new_position):
            self.position = new_position

    def possible_moves(self, is_valid_move):
        moves = []
        for direction in range(4):
            new_position = self._get_new_position(direction)
            if is_valid_move(new_position):
                moves.append((direction, new_position))
        return moves

    def _get_new_position(self, direction):
        new_position = self.position.copy()
        if direction == 0:   # up
            new_position[0] -= 1
        elif direction == 1: # right
            new_position[1] += 1
        elif direction == 2: # down
            new_position[0] += 1
        elif direction == 3: # left
            new_position[1] -= 1
        return new_position

class Chaser:
    def __init__(self, position):
        self.position = position

    def move_towards_runner(self, runner_position, is_valid_move):
        move_direction = np.argmax(np.abs(runner_position - self.position))
        direction_sign = np.sign(runner_position[move_direction] - self.position[move_direction])
        new_position = self.position.copy()
        new_position[move_direction] += direction_sign
        if is_valid_move(new_position):
            self.position = new_position

    def possible_moves(self, is_valid_move):
        moves = []
        for direction in range(4):
            new_position = self._get_new_position(direction)
            if is_valid_move(new_position):
                moves.append((direction, new_position))
        return moves

    def _get_new_position(self, direction):
        new_position = self.position.copy()
        if direction == 0:   # up
            new_position[0] -= 1
        elif direction == 1: # right
            new_position[1] += 1
        elif direction == 2: # down
            new_position[0] += 1
        elif direction == 3: # left
            new_position[1] -= 1
        return new_position

class Obstacle:
    def __init__(self, position):
        self.position = position

class ChaseEnvironment(py_environment.PyEnvironment):
    def __init__(self, grid_size, num_obstacles):
        self._grid_size = grid_size
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(grid_size, grid_size, 3), dtype=np.float32, minimum=0., maximum=1., name='observation')
        self._runner = Runner(self._init_position())
        self._chaser = Chaser(self._init_position())
        self._obstacles = [Obstacle(self._init_position()) for _ in range(num_obstacles)]

        # Define the action distribution spec
        self._action_distribution_spec = tfp.distributions.Categorical(logits=tf.zeros((self._action_spec.maximum + 1,)))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _init_position(self):
        return np.random.randint(0, self._grid_size, size=(2,))
    
    def _reset_position(self):
        position = self._init_position()
        while not self.is_valid_position(position):
            position = self._init_position()
        return position
    
    def is_valid_position(self, position):
        is_inside_grid = 0 <= position[0] < self._grid_size and 0 <= position[1] < self._grid_size
        not_occupied =  not any(np.array_equal(position, obs.position) for obs in self._obstacles)
                        # not np.array_equal(position, self._runner.position) and \
                        # not np.array_equal(position, self._chaser.position)

        return is_inside_grid and not_occupied

    def get_current_state(self):
        return self._state

    def _set_state(self, state):
        self._runner.position = np.array(np.where(state[:, :, 0] == 1.)).flatten()
        self._chaser.position = np.array(np.where(state[:, :, 1] == 1.)).flatten()
        self._obstacles = []
        for i in range(2, state.shape[2]):
            self._obstacles.append(Obstacle(np.array(np.where(state[:, :, i] == 1.)).flatten()))
        self._state = state

    def _create_state(self):
        state = np.zeros((self._grid_size, self._grid_size, 3), dtype=np.float32)
        state[tuple(self._runner.position), 0] = 1.
        state[tuple(self._chaser.position), 1] = 1.
        for obs in self._obstacles:
            state[tuple(obs.position), 2] = 1.
        return state

    def _is_done(self):
        return np.array_equal(self._runner.position, self._chaser.position)

    def _reward(self):
        return np.linalg.norm(self._runner.position - self._chaser.position)
        # return -np.linalg.norm(self._runner_position - self._chaser_position) for chaser
        # return 1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for runner for smoothly approach
        # return -1 / (1 + np.linalg.norm(self._runner_position - self._chaser_position)) for chaser for smoothly avoid

    def _reset(self):
        self._runner.position = self._reset_position()
        self._chaser.position = self._reset_position()
        self._state = self._create_state()
        return ts.restart(self._state)

    def _step(self, action):
        if self._is_done():
            return self._reset()

        self._runner.move(action, self.is_valid_position)

        # The chaser moves towards the runner.
        self._chaser.move_towards_runner(self._runner.position, self.is_valid_position)
        self._state = self._create_state()

        if self._is_done():
            return ts.termination(self._state, self._reward())
        else:
            return ts.transition(self._state, reward=self._reward(), discount=0.999)
        
    def _render(self):
        # if there is no screen, create one
        if not hasattr(self, '_screen'):
            # Create the display surface
            cell_size = 50
            width, height = self._grid_size * cell_size, self._grid_size * cell_size
            self.screen = pygame.display.set_mode((width, height))
            self.cell_size = cell_size

            # Create the grid rects
            self.grid_rects = [pygame.Rect(i, j, cell_size, cell_size) for j in range(0, height, cell_size) for i in range(0, width, cell_size)]

            # Create runner, chaser, and obstacle rects
            self.runner_rect = pygame.Rect(0, 0, cell_size, cell_size)
            self.chaser_rect = pygame.Rect(0, 0, cell_size, cell_size)
            self.obstacle_rects = [pygame.Rect(0, 0, cell_size, cell_size) for _ in range(len(self._obstacles))]
        
        self.screen.fill((0, 0, 0))

        # Draw the grid
        for rect in self.grid_rects:
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Update and draw the runner (blue) and the chaser (red)
        self.runner_rect.topleft = (self._runner.position[1] * self.cell_size, self._runner.position[0] * self.cell_size)
        self.chaser_rect.topleft = (self._chaser.position[1] * self.cell_size, self._chaser.position[0] * self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), self.runner_rect)
        pygame.draw.rect(self.screen, (255, 0, 0), self.chaser_rect)

        # Update and draw the obstacles (gray)
        for i, obstacle in enumerate(self._obstacles):
            self.obstacle_rects[i].topleft = (obstacle.position[1] * self.cell_size, obstacle.position[0] * self.cell_size)
            pygame.draw.rect(self.screen, (128, 128, 128), self.obstacle_rects[i])

        pygame.display.flip()

    def copy(self):
        env = ChaseEnvironment(self._grid_size, len(self._obstacles))
        env._runner.position = self._runner.position.copy()
        env._chaser.position = self._chaser.position.copy()
        for i, obstacle in enumerate(self._obstacles):
            env._obstacles[i].position = obstacle.position.copy()
        return env

class TreeNode:
    def __init__(self, positions, value=None):
        self.positions = positions
        self.value = value
        self.children = []

class MinimaxAlphaBetaAgent:
    def __init__(self, depth):
        self.depth = depth
        self.tree_nodes = {}

    def save_current_state(self, player):
        return player.position.copy()

    def restore_state(self, player, saved_position):
        player.position = saved_position

    def get_move_and_new_position(self, player, environment):
        for direction, new_position in player.possible_moves(environment.is_valid_position):
            saved_position = self.save_current_state(player)
            player.position = new_position
            yield direction, new_position, saved_position

    def process_maximizing_player(self, runner, chaser, environment, depth, alpha, beta, tree_nodes):
        max_value = -float('inf')
        for _, new_position, saved_position in self.get_move_and_new_position(runner, environment):
            value = self.minimax_alpha_beta(runner, chaser, environment, depth - 1, alpha, beta, False, tree_nodes)
            self.restore_state(runner, saved_position)
            max_value = max(max_value, value)
            alpha = max(alpha, max_value)
            if beta <= alpha:
                break
        return max_value

    def process_minimizing_player(self, runner, chaser, environment, depth, alpha, beta, tree_nodes):
        min_value = float('inf')
        for _, new_position, saved_position in self.get_move_and_new_position(chaser, environment):
            value = self.minimax_alpha_beta(runner, chaser, environment, depth - 1, alpha, beta, True, tree_nodes)
            self.restore_state(chaser, saved_position)
            min_value = min(min_value, value)
            beta = min(beta, min_value)
            if beta <= alpha:
                break
        return min_value

    def minimax_alpha_beta(self, runner, chaser, environment, depth, alpha, beta, maximizing_runner, tree_nodes):
        current_positions = (tuple(runner.position.copy()), tuple(chaser.position.copy()))
        key = str(current_positions)  # Convert tuple to a hashable string key

        if key in tree_nodes:
            return tree_nodes[key].value

        if depth == 0 or environment._is_done():
            value = environment._reward()
            tree_nodes[key] = TreeNode(current_positions, value)
            return value

        if maximizing_runner:
            value = self.process_maximizing_player(runner, chaser, environment, depth, alpha, beta, tree_nodes)
        else:
            value = self.process_minimizing_player(runner, chaser, environment, depth, alpha, beta, tree_nodes)

        tree_nodes[key] = TreeNode(current_positions, value)
        return value

    # Prioritize moves method remains the same
    def prioritize_moves(self, runner, chaser, environment, moves):
        ordered_moves = []

        for _, move in moves:
            distance = np.linalg.norm(move - chaser.position)
            ordered_moves.append((distance, move))

        ordered_moves.sort(key=lambda x: x[0], reverse=True)

        return ordered_moves

    def best_move_for_runner(self, runner, chaser, environment, depth, tree_nodes):
        max_value = -float('inf')
        best_move = None
        for direction, new_position, saved_position in self.get_move_and_new_position(runner, environment):
            value = self.minimax_alpha_beta(runner, chaser, environment, depth - 1, -float('inf'), float('inf'), False, tree_nodes)
            self.restore_state(runner, saved_position)
            if value > max_value:
                max_value = value
                best_move = direction
        return best_move

    def best_move_for_chaser(self, runner, chaser, environment, depth, tree_nodes):
        min_value = float('inf')
        best_move = None
        for direction, new_position, saved_position in self.get_move_and_new_position(chaser, environment):
            value = self.minimax_alpha_beta(runner, chaser, environment, depth - 1, -float('inf'), float('inf'), True, tree_nodes)
            self.restore_state(chaser, saved_position)
            if value < min_value:
                min_value = value
                best_move = direction
        return best_move

class MCTSNode:
    def __init__(self, state, parent=None, reward=0, num_visits=0):
        self.state = state
        self.parent = parent
        self.reward = reward
        self.num_visits = num_visits
        self.children = {}

class MCTSAgent:
    def __init__(self, num_simulations=1000, exploration_weight=1.):
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def uct_value(self, parent_visit, child_reward, child_visit):
        # Calculate the UCT value
        return child_reward / (child_visit + 1e-5) + self.exploration_weight * \
               np.sqrt(np.log(parent_visit + 1) / (child_visit + 1e-5))

    def select_node(self, node):
        # Select the node with the highest UCT value
        best_node = max(node.children.items(), 
                        key=lambda item: self.uct_value(node.num_visits, item[1].reward, item[1].num_visits))

        return best_node

    def expand_node(self, node, action):
        # Expand the node with the given action
        new_state = node.state.copy()
        new_state._step(action)
        child_node = MCTSNode(new_state, parent=node)
        node.children[action] = child_node
        return child_node

    def simulate(self, node):
        # Simulate a random game from the given node until a terminal state is reached
        while not node.state._is_done():
            action = self.get_random_action(node)
            node = self.expand_node(node, action)
        return node.state._reward()

    def backpropagate(self, node, reward):
        # Update the current node and all its ancestors with the result of the simulation
        node.num_visits += 1
        node.reward += reward
        if node.parent is not None:
            self.backpropagate(node.parent, reward)

    def get_random_action(self, node):
        possible_moves = node.state._runner.possible_moves(node.state.is_valid_position)
        direction = list(zip(*possible_moves))[0]
        return np.random.choice(direction)

    def get_best_action(self, root):
        # Return the action leading to the child with the highest number of visits
        return max(root.children.items(), key=lambda item: item[1].num_visits)[0]

    def plan(self, state):
        # Create the root node
        root = MCTSNode(state.copy())

        # Perform MCTS
        for _ in range(self.num_simulations):
            node = root
            while node.children:
                _, node = self.select_node(node)
            if not node.state._is_done():
                node = self.expand_node(node, self.get_random_action(node))
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        # Return the best action
        return self.get_best_action(root)


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

class ResidualBlock(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.add = layers.Add()
        self.relu2 = layers.Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        shortcut = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([shortcut, x])
        x = self.relu2(x)
        return x
    
class AlphaZeroNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.input_layer = layers.InputLayer(input_shape=input_shape)

        self.initial_conv = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])

        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)

        # policy head
        self.policy_head = tf.keras.Sequential([
            layers.Conv2D(2, 1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(num_actions, activation="softmax")
        ])

        # value head
        self.value_head = tf.keras.Sequential([
            layers.Conv2D(1, 1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="tanh")
        ])

        self.compile_model()

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def compile_model(self):
        self.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                     loss=['categorical_crossentropy', 'mse'],
                     loss_weights=[1.0, 1.0])

class AlphaZeroAgent:
    def __init__(self, grid_size, num_actions, num_simulations=1000, exploration_weight=1.):
        self.network = AlphaZeroNetwork(grid_size, num_actions)
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.mcts_agent = MCTSAgent(num_simulations, exploration_weight)

    def get_action(self, state):
        # Plan the action using MCTS
        action = self.mcts_agent.plan(state)

        # Get the network's prediction
        state_tensor = tf.convert_to_tensor(state._create_state())
        policy, value = self.network(state_tensor[None, ...])
        network_action = np.argmax(policy)

        # Balance between the network's prediction and MCTS
        action = network_action if np.random.uniform() < 0.5 else action
        return action

    def train(self, state, action, reward):
        # Convert to tensors
        state_tensor = tf.convert_to_tensor(state)
        action_tensor = tf.convert_to_tensor(action)
        reward_tensor = tf.convert_to_tensor(reward)

        # Calculate loss
        with tf.GradientTape() as tape:
            policy, value = self.network(state_tensor[None, ...])
            policy_loss = tf.keras.losses.sparse_categorical_crossentropy(action_tensor[None, ...], policy)
            value_loss = tf.keras.losses.mean_squared_error(reward_tensor[None, ...], value)
            loss = policy_loss + value_loss

        # Get gradients
        gradients = tape.gradient(loss, self.network.trainable_variables)

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam()

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))


env = ChaseEnvironment(grid_size=8, num_obstacles=5)
env.reset()
env._render()

# Create the agent
# agent = MinimaxAlphaBeta(depth=3)
# agent = MCTSAgent(num_simulations=10)
agent = AlphaZeroAgent(grid_size=8, num_actions=4, num_simulations=10)

# Run the environment
pygame.init()
running = True
num_episodes = 10
for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.is_last() and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # best_runner_move = agent.best_move_for_runner(env._runner, env._chaser, env, agent.depth, agent.tree_nodes)
        best_runner_move = agent.get_action(env)
        time_step = env.step(best_runner_move)
        env._render()
        pygame.time.delay(500)
    print('Episode ended! Reward: {}'.format(time_step.reward))
    pygame.time.delay(100)
pygame.quit()


"""
The code implementation for AlphaZero and Monte Carlo Tree Search (MCTS) seems conceptually correct in structure. However, there are a few things missing or potentially wrong in the implementation:

1. The `AlphaZeroAgent`'s `train` method should ideally be updated to account for batches of experiences rather than single instances for more effective training. Reinforcement learning typically benefits from experience replay, where the agent trains on batches of previous experiences.

2. In the `AlphaZeroAgent`'s `train` method, it seems that the gradients are calculated based on a single example (`state`). However, the `state` passed in the `train` function should ideally be a batch of states from the replay buffer. It would be beneficial to check that this is the case in the calling code.

3. In the AlphaZero paper, the MCTS uses the policy output of the network to guide the exploration process, but in your code, the `MCTSAgent` uses a uniform random policy. The `plan` method should ideally use the output of the neural network policy to guide the tree expansion.

4. In the `AlphaZeroAgent`'s `get_action` method, you are mixing actions between MCTS and the policy network randomly. However, it's supposed to be MCTS that uses the policy network's output for more informed exploration, and finally, the action is selected based on MCTS output.

5. It would be ideal to separate the concerns of the agent and environment. The `MCTSAgent` and `AlphaZeroAgent` should not directly manipulate the state of the environment. They should return actions, and the environment should change its state based on those actions.

6. Training the `AlphaZeroNetwork` could potentially benefit from including an entropy term to encourage exploration. In your current implementation, it doesn't look like you're using entropy in your loss function.
"""
