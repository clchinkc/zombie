
import numpy as np


class Environment:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.transitions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.costs = {'S': 1, 'G': 0, 'X': 0, 'T': 1}

    def is_valid(self, row, col):
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols and self.grid[row][col] != 'X'

    def get_next_state(self, state, action):
        row, col = state
        drow, dcol = self.transitions[action]
        new_row, new_col = row + drow, col + dcol

        if self.is_valid(new_row, new_col):
            return (new_row, new_col)
        else:
            return state

    def get_cost(self, state):
        row, col = state
        return self.costs[self.grid[row][col]]

    def get_actions(self, state):
        return list(self.transitions.keys())


def compute_action_values(env, state: tuple[int, int], value_function: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute the action values for a state using the value function."""
    if env.grid[state[0]][state[1]] in ('X', 'G'):
        return np.array([])

    action_values = np.zeros(len(env.transitions))

    for i, action in enumerate(env.transitions):
        next_state = env.get_next_state(state, action)
        next_reward = env.get_cost(next_state) + gamma * value_function[next_state]
        action_values[i] = next_reward

    return action_values

def update_value_function(env, gamma: float, state: tuple[int, int], value_function: np.ndarray) -> float:
    """Update the value of a state using the Bellman equation."""
    current_value = value_function[state]
    action_values = compute_action_values(env, state, value_function, gamma)
    
    if len(action_values) > 0:
        value_function[state] = np.min(action_values)

    return abs(current_value - value_function[state])

def select_best_action(env, state: tuple[int, int], value_function: np.ndarray) -> str:
    """Select the best action for a state using the value function."""
    action_values = compute_action_values(env, state, value_function)
    
    if len(action_values) == 0:
        return ""

    return list(env.transitions.keys())[np.argmin(action_values)]


def value_iteration(env, gamma: float=0.99, epsilon: float=1e-5) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the value iteration algorithm to find the optimal value function and policy.

    Args:
        env: The environment object that defines the MDP.
        gamma: The discount factor for future rewards.
        epsilon: The convergence threshold for the algorithm.

    Returns:
        A tuple (value_function, policy) of numpy arrays representing the optimal value function
        and policy, respectively.
    """
    # Initialize the value function and policy arrays
    value_function = np.zeros((env.n_rows, env.n_cols))
    policy = np.empty((env.n_rows, env.n_cols), dtype='<U1')

    while True:
        max_change = 0

        # Update the value function for each state
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                state = (row, col)

                change = update_value_function(env, gamma, state, value_function)
                max_change = max(max_change, change)

                # Select the best action for the state
                policy[state] = select_best_action(env, state, value_function)

        if max_change < epsilon:
            # The algorithm has converged, return the optimal value function and policy
            return value_function, policy


def policy_evaluation(env, policy, V, gamma=0.99, epsilon=1e-5):
    while True:
        delta = 0

        for row in range(env.n_rows):
            for col in range(env.n_cols):
                state = (row, col)
                if env.grid[row][col] in ('X', 'G'):
                    continue

                v = V[state]
                action = policy[state]
                next_state = env.get_next_state(state, action)
                next_reward = env.get_cost(next_state) + gamma * V[next_state]

                V[state] = next_reward
                delta = max(delta, abs(v - V[state]))

        if delta < epsilon:
            break

    return V

def policy_improvement(env, policy, V, gamma=0.99):
    stable = True

    for row in range(env.n_rows):
        for col in range(env.n_cols):
            state = (row, col)
            if env.grid[row][col] in ('X', 'G'):
                continue

            old_action = policy[state]
            values = []

            for action in env.get_actions(state):
                next_state = env.get_next_state(state, action)
                next_reward = env.get_cost(next_state) + gamma * V[next_state]
                values.append(next_reward)

            policy[state] = env.get_actions(state)[np.argmin(values)]

            if old_action != policy[state]:
                stable = False

    return stable, policy

def policy_iteration(env, gamma=0.99, epsilon=1e-5):
    V = np.zeros((env.n_rows, env.n_cols))
    policy = np.random.choice(list(env.transitions.keys()), (env.n_rows, env.n_cols))
    policy = np.array([['' if env.grid[row][col] in ('X', 'G') else policy[row, col] for col in range(env.n_cols)] for row in range(env.n_rows)])

    while True:
        V = policy_evaluation(env, policy, V, gamma, epsilon)
        stable, policy = policy_improvement(env, policy, V, gamma)

        if stable:
            break

    return V, policy



def print_map(env):
    color_map = {'S': '\033[92mS\033[0m', 'G': '\033[94mG\033[0m', 'X': '\033[91mX\033[0m', 'T': '\033[93mT\033[0m'}
    
    for row in env.grid:
        print(' '.join([color_map[cell] for cell in row]))

def printV(V):
    for row in range(V.shape[0]):
        for col in range(V.shape[1]):
            print(f"{V[row, col]:.2f}", end=' ')
        print()

def printPolicy(policy):
    for row in range(policy.shape[0]):
        for col in range(policy.shape[1]):
            if policy[row, col] == 'U':
                print('\u2191', end=' ')
            elif policy[row, col] == 'D':
                print('\u2193', end=' ')
            elif policy[row, col] == 'L':
                print('\u2190', end=' ')
            elif policy[row, col] == 'R':
                print('\u2192', end=' ')
            else:
                print('-', end=' ')
        print()

grid = [
    ['S', 'T', 'X', 'T', 'G'],
    ['T', 'X', 'T', 'T', 'T'],
    ['T', 'T', 'T', 'X', 'T'],
    ['X', 'T', 'X', 'T', 'T'],
    ['T', 'T', 'T', 'T', 'T']
]

env = Environment(grid)
print_map(env)


# Testing the value iteration
# V_value_iter, policy_value_iter = value_iteration(env)
# print("\nState Values (Value Iteration):")
# printV(V_value_iter)

# print("\nOptimal Policy (Value Iteration):")
# printPolicy(policy_value_iter)


# Testing the policy iteration
# V_policy_iter, policy_policy_iter = policy_iteration(env)
# print("\nState Values (Policy Iteration):")
# printV(V_policy_iter)

# print("\nOptimal Policy (Policy Iteration):")
# printPolicy(policy_policy_iter)

print("\n")

class Agent:
    def __init__(self, env, gamma=0.99, epsilon=1e-5, algorithm='value_iteration'):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.algorithm = algorithm
        
        if self.algorithm == 'value_iteration':
            self.V, self.policy = value_iteration(env, gamma, epsilon)
        elif self.algorithm == 'policy_iteration':
            self.V, self.policy = policy_iteration(env, gamma, epsilon)
        else:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
    
    def move(self, state):
        return self.env.get_next_state(state, self.policy[state])

grid = [
    ['S', 'T', 'X', 'T', 'G'],
    ['T', 'X', 'T', 'T', 'T'],
    ['T', 'T', 'T', 'X', 'T'],
    ['X', 'T', 'X', 'T', 'T'],
    ['T', 'T', 'T', 'T', 'T']
]

env = Environment(grid)
agent = Agent(env, algorithm='policy_iteration')

current_state = (0, 0)
print(f"Current state: {current_state}")

next_state = agent.move(current_state)
print(f"Next state: {next_state}")
print(f"Cost: {env.get_cost(next_state)}")

next_state = agent.move(next_state)
print(f"Next state: {next_state}")
print(f"Cost: {env.get_cost(next_state)}")

next_state = agent.move(next_state)
print(f"Next state: {next_state}")
print(f"Cost: {env.get_cost(next_state)}")


# https://www.datascienceblog.net/post/reinforcement-learning/mdps_dynamic_programming/
