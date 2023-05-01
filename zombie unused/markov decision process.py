
"""
The code is an implementation of two reinforcement learning algorithms: value iteration and policy iteration. It uses a 3x4 grid world as the environment and aims to find the optimal policy for navigating the environment.

The code first defines a function printEnvironment that prints the current state of the environment in a formatted manner. The environment is represented as a 2D list of utilities (U) or actions (policy). The function prints the utilities or actions in a grid format with "|" and "-" characters to separate the cells.

The code also defines two helper functions getU and calculateU. getU takes as input the current state (r, c), an action (0-3), and the current utilities (U) and returns the utility of the state reached by performing the given action. calculateU takes as input the current state (r, c), an action (0-3), and the current utilities (U) and calculates the utility of the state that would be reached by performing the given action.

The main algorithms are implemented in two functions: valueIteration and policyIteration. valueIteration performs value iteration until the utilities converge. It iteratively updates the utilities using the Bellman update equation and stops when the error between the current and next utilities is less than a given threshold. getOptimalPolicy is then called to get the optimal policy based on the converged utilities.

policyIteration performs policy iteration by repeatedly improving the current policy until it converges. It starts by initializing a random policy and then iteratively updates the utilities using the simplified Bellman update equation, and the policy using the action that maximizes the utility for each state. It stops when the policy is unchanged after an iteration.

The main function main takes as input the initial utilities (U), the initial policy, and the chosen algorithm (either 'value_iteration' or 'policy_iteration'). It calls the corresponding algorithm function and prints the results.

The code also sets up some constants and variables needed for the environment and reinforcement learning algorithms. These include the reward for non-terminal states, the discount factor, the maximum allowable error, the actions that can be taken, and the dimensions of the environment. A random policy is also initialized in policy.
"""

import random


# Visualization
def printEnvironment(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if r == c == 1:
                val = "WALL"
            elif r <= 1 and c == 3:
                val = "+1" if r == 0 else "-1"
            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (newR == newC == 1): # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u


# Value iteration

# Perform value iteration until the utilities converge
def valueIteration(U):
    print("During the value iteration:\n")
    while True:
        nextU = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        printEnvironment(U)
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

# Get the optimal policy from U
def getOptimalPolicy(U):
    policy = [[-1, -1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if (r <= 1 and c == 3) or (r == c == 1):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy




# Policy iteration

# Perform some simplified value iteration steps to get an approximation of the utilities
def policyEvaluation(policy, U):
    while True:
        nextU = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                nextU[r][c] = calculateU(U, r, c, policy[r][c]) # simplified Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

def policyIteration(policy, U):
    print("During the policy iteration:\n")
    while True:
        U = policyEvaluation(policy, U)
        unchanged = True
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == 3) or (r == c == 1):
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculateU(U, r, c, action)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculateU(U, r, c, policy[r][c]):
                    policy[r][c] = maxAction # the action that maximizes the utility
                    unchanged = False
        if unchanged:
            break
        printEnvironment(policy)
    return policy

def main(U, policy, method='value_iteration'):
    # Print the initial environment
    print("The initial U is:\n")
    printEnvironment(U)

    if method == 'value_iteration':
        # Value iteration
        U = valueIteration(U)

        # Get the optimal policy from U and print it
        policy = getOptimalPolicy(U)
        print("The optimal policy is:\n")
        printEnvironment(policy, True)
    elif method == 'policy_iteration':
        # Print the initial random policy
        print("The initial random policy is:\n")
        printEnvironment(policy, True)

        # Policy iteration
        policy = policyIteration(policy, U)

        # Print the optimal policy
        print("The optimal policy is:\n")
        printEnvironment(policy, True)
    else:
        print("Invalid method. Choose either 'value_iteration' or 'policy_iteration'.")


if __name__ == '__main__':
    
    # Arguments
    REWARD = -0.01 # constant reward for non-terminal states
    DISCOUNT = 0.99
    MAX_ERROR = 10**(-3)

    # Set up the initial environment
    NUM_ACTIONS = 4
    ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
    NUM_ROW = 3
    NUM_COL = 4
    U = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
    policy = [[random.randint(0, 3) for j in range(NUM_COL)] for i in range(NUM_ROW)] # construct a random policy
    
    method = 'value_iteration'
    # method = 'policy_iteration'
    main(U, policy, method)
