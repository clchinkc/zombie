

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


class TreeNode:
    def __init__(self, positions, value=None):
        self.positions = positions
        self.value = value
        self.children = []

def minimax_alpha_beta(runner, chaser, environment, depth, alpha, beta, maximizing_runner, tree_nodes):
    current_positions = (tuple(runner.position.copy()), tuple(chaser.position.copy()))
    key = str(current_positions)  # Convert tuple to a hashable string key

    if key in tree_nodes:
        return tree_nodes[key].value

    if depth == 0 or environment._is_done():
        value = environment._reward()
        tree_nodes[key] = TreeNode(current_positions, value)
        return value

    if maximizing_runner:
        max_value = -float('inf')

        # Reorder moves based on a heuristic for move ordering
        moves = runner.possible_moves(environment.is_valid_position)
        ordered_moves = prioritize_moves(runner, chaser, environment, moves)

        for _, new_position in ordered_moves:
            saved_position = runner.position.copy()
            runner.position = new_position
            value = minimax_alpha_beta(runner, chaser, environment, depth - 1, alpha, beta, False, tree_nodes)
            runner.position = saved_position
            max_value = max(max_value, value)
            alpha = max(alpha, max_value)
            if beta <= alpha:
                break

        tree_nodes[key] = TreeNode(current_positions, max_value)
        return max_value
    else:
        min_value = float('inf')

        # Reorder moves based on a heuristic for move ordering
        moves = chaser.possible_moves(environment.is_valid_position)
        ordered_moves = prioritize_moves(runner, chaser, environment, moves)

        for _, new_position in ordered_moves:
            saved_position = chaser.position.copy()
            chaser.position = new_position
            value = minimax_alpha_beta(runner, chaser, environment, depth - 1, alpha, beta, True, tree_nodes)
            chaser.position = saved_position
            min_value = min(min_value, value)
            beta = min(beta, min_value)
            if beta <= alpha:
                break

        tree_nodes[key] = TreeNode(current_positions, min_value)
        return min_value

def prioritize_moves(runner, chaser, environment, moves):
    ordered_moves = []

    for _, move in moves:
        distance = np.linalg.norm(move - chaser.position)
        ordered_moves.append((distance, move))

    ordered_moves.sort(key=lambda x: x[0], reverse=True)

    return ordered_moves

def best_move_for_runner(runner, chaser, environment, depth, tree_nodes):
    max_value = -float('inf')
    best_move = None
    for direction, new_position in runner.possible_moves(environment.is_valid_position):
        saved_position = runner.position.copy()
        runner.position = new_position
        value = minimax_alpha_beta(runner, chaser, environment, depth - 1, -float('inf'), float('inf'), False, tree_nodes)
        runner.position = saved_position
        if value > max_value:
            max_value = value
            best_move = direction
    return best_move

def best_move_for_chaser(runner, chaser, environment, depth, tree_nodes):
    min_value = float('inf')
    best_move = None
    for direction, new_position in chaser.possible_moves(environment.is_valid_position):
        saved_position = chaser.position.copy()
        chaser.position = new_position
        value = minimax_alpha_beta(runner, chaser, environment, depth - 1, -float('inf'), float('inf'), True, tree_nodes)
        chaser.position = saved_position
        if value < min_value:
            min_value = value
            best_move = direction
    return best_move

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
        self._state = self._create_state()
        self.tree_nodes = {}

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

        best_runner_move = best_move_for_runner(self._runner, self._chaser, self, depth=5, tree_nodes=self.tree_nodes)
        self._runner.move(best_runner_move, self.is_valid_position)

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
            self.obstacle_rects = [pygame.Rect(0, 0, cell_size, cell_size) for _ in range(num_obstacles)]
        
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

class Node:
    
        def __init__(self, parent=None, action=None, reward=0):
            self.parent = parent
            self.action = action
            self.reward = reward
            self.children = []
    
        def expand(self):
            new_node = Node(parent=self)
            self.children.append(new_node)
            return new_node
    
        def is_terminal(self):
            return len(self.children) == 0
    
        def select_best_action(self):
            best_action = None
            best_reward = float('-inf')
            for child in self.children:
                if child.reward > best_reward:
                    best_reward = child.reward
                    best_action = child.action
            return best_action
    
        def select_leaf_node(self):
            while not self.is_terminal():
                self.select_best_child()
            return self
        
        def select_best_child(self):
            best_child = None
            best_score = float('-inf')
            for child in self.children:
                score = child.reward + math.sqrt(2 * math.log(self.get_num_visits()) / 1 + child.get_num_visits())
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child
        
        def get_num_visits(self):
            num_visits = 0
            while self is not None:
                num_visits += 1
                self = self.parent
            return num_visits

class MCTS():

    def __init__(self, num_iterations=1000, c=1.0):
        self.num_iterations = num_iterations
        self.c = c

        # Initialize the MCTS tree.
        self.tree = Node()

    def act(self, state):
        # Perform MCTS.
        for _ in range(self.num_iterations):
            reward = self._select_expand_and_simulate(state)

        # Select the best action according to the tree.
        best_action = self.tree.select_best_action()

        return best_action

    def _select_expand_and_simulate(self, state):
        # Select a leaf node.
        node = self.tree.select_leaf_node()

        # Expand the node.
        if not node.is_terminal():
            node = node.expand()

        # Simulate a game from the node.
        reward = self._simulate_game(node)

        # Backpropagate the reward to the tree.
        self._backpropagate(node, reward)

        return reward

    def _simulate_game(self, node):
        # If the game is over, return the reward.
        if node.is_terminal():
            return node.reward

        # Select the next action according to the tree.
        action = node.select_best_action()

        # Simulate the next state and reward.
        next_state, reward = self._simulate_game(node.children[action])

        # Return the reward from the simulation.
        return reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.reward += reward * self.c
            node = node.parent



class MCTSChaseAgent():

    def __init__(self, grid_size, num_iterations=1000, c=1.0):
        self.num_iterations = num_iterations
        self.c = c

        # Initialize the MCTS tree.
        self.tree = Node()

        # Initialize the neural network.
        self.model = create_neural_network((grid_size, grid_size, 2), 4) # with policy and value heads

        # Initialize the hyperparameters.
        self.epsilon = 0.1

    def act(self, state):
        # Perform MCTS.
        for _ in range(self.num_iterations):
            reward = self._select_expand_and_simulate(state)

        # Select the best action according to the tree.
        best_action = self.tree.select_best_action()

        return best_action

    def _select_expand_and_simulate(self, state):
        # Select a leaf node.
        node = self.tree.select_leaf_node()

        # Expand the node.
        if not node.is_terminal():
            node = node.expand()

        # Simulate a game from the node.
        reward = self._simulate_game(node)

        # Backpropagate the reward to the tree.
        self._backpropagate(node, reward)

        return reward

    def _simulate_game(self, node):
        # If the game is over, return the reward.
        if node.is_terminal():
            return node.reward

        # Select the next action according to the policy network.
        action_probs, _ = self.model.predict(np.array([node.state]))
        action = np.argmax(action_probs[0])

        # Simulate the next state and reward.
        next_state, reward = self._simulate_game(node.children[action])

        # Return the next state and reward.
        return next_state, reward

    def _backpropagate(self, node, reward):
        # Update the node's reward.
        node.reward += reward

        # If the node is not the root, backpropagate.
        if node.parent is not None:
            self._backpropagate(node.parent, reward)

num_obstacles = 5
grid_size = 10
env = ChaseEnvironment(grid_size, num_obstacles)
env.reset()
env._render()

# Create the MCTS agent.
mcts = MCTSChaseAgent(grid_size=grid_size, num_iterations=10)

# Run the environment
num_episodes = 10
for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.is_last():
        # use minimax to find the best move for the runner
        # best_runner_move = best_move_for_runner(env._runner, env._chaser, env, depth=5, tree_nodes=env.tree_nodes)
        best_runner_move = mcts.act(env._state)
        time_step = env.step(best_runner_move)
        env._render()
        pygame.time.delay(500)
    print('Episode ended! Reward: {}'.format(time_step.reward))
    pygame.time.delay(1000)
pygame.quit()



