
"""
The code defines an environment and performs a value iteration algorithm to find the optimal policy for the given environment. The environment is represented as a grid, where each cell can contain one of the following: 'S' (starting point), 'G' (goal), 'X' (obstacle), or 'T' (transitional cell). The transitional cell has a cost of 1, while all other cells have a cost of 0. The objective of the algorithm is to find the optimal path from the starting point to the goal, while avoiding obstacles.

The Environment class defines the environment, with methods to check the validity of a state (i.e., whether it is within the grid and not an obstacle), get the next state given an action, get the cost of a state, and get the possible actions for a state. The value_iteration function performs the value iteration algorithm, which involves iteratively updating the state values (i.e., the expected reward from a state), until convergence (i.e., the difference between the new state values and the old state values is below a certain threshold). The optimal policy (i.e., the optimal action to take in each state) is then determined based on the updated state values.

The print_map, printV, and printPolicy functions are helper functions to print the environment, the state values, and the optimal policy, respectively.

Overall, the code represents a simple implementation of value iteration for solving a grid-world problem, where the objective is to find the optimal path from a starting point to a goal, while avoiding obstacles.
"""

import numpy as np


class Environment:
    def __init__(self, grid):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.transitions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.costs = {'S': 0, 'G': 0, 'X': 0, 'T': 1}

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

def value_iteration(env, gamma=0.99, epsilon=1e-4):
    V = np.zeros((env.n_rows, env.n_cols))
    policy = np.empty((env.n_rows, env.n_cols), dtype='<U1')

    while True:
        delta = 0

        for row in range(env.n_rows):
            for col in range(env.n_cols):
                state = (row, col)
                if env.grid[row][col] in ('X', 'G'):
                    continue

                v = V[state]
                values = []
                for action in env.get_actions(state):
                    next_state = env.get_next_state(state, action)
                    next_reward = env.get_cost(next_state) + gamma * V[next_state]
                    values.append(next_reward)

                V[state] = min(values)
                policy[state] = env.get_actions(state)[np.argmin(values)]
                delta = max(delta, abs(v - V[state]))

        if delta < epsilon:
            break

    return V, policy

def print_map(env):
    for row in range(env.n_rows):
        for col in range(env.n_cols):
            cell = env.grid[row][col]
            if cell == 'S':
                print('\033[92mS\033[0m', end=' ')
            elif cell == 'G':
                print('\033[94mG\033[0m', end=' ')
            elif cell == 'X':
                print('\033[91mX\033[0m', end=' ')
            elif cell == 'T':
                print('\033[93mT\033[0m', end=' ')
        print()

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

V, policy = value_iteration(env)
print("\nState Values:")
printV(V)

print("\nOptimal Policy:")
printPolicy(policy)


