import numpy as np
import matplotlib.pyplot as plt

# define the possible states as a list
states = ["safe_house", "zombie_nearby", "resources"]


# define the possible actions as a list
actions = ["move", "stay", "fight"]

# define the rewards
KILL_ZOMBIE = 10
DIE_BY_ZOMBIE = -100
SURVIVE = 1
FIND_RESOURCES = 5
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.999

# define the number of states and actions
num_states = len(states)
num_actions = len(actions)

# initialize the Q-table with all zeros
Q = np.zeros((num_states, num_actions))

# define the reinforcement learning algorithm
def q_learning(Q, state, action, reward, next_state):
    # calculate the maximum expected reward for the next state
    state_index = states.index(state)
    action_index = actions.index(action)
    next_state_index = states.index(next_state)
    max_reward = max(Q[next_state_index][:])

    # update the Q-value for the current state and action
    Q[state_index][action_index] = (1 - LEARNING_RATE) * Q[state_index][action_index] + \
        LEARNING_RATE * (reward + DISCOUNT_RATE * max_reward)

# define the function for checking the end condition
def end_condition(reward):
    # end the simulation if the person is killed by a zombie
    if reward == DIE_BY_ZOMBIE:
        return True
    else:
        return False
    
# define the function for executing an action
def execute_action(state, action):
    print(f"state: {state}, action: {action}")
    if action == "stay":
        # stay in the current location and update the state
        next_state = state
        if state == "zombie_nearby":
            reward = DIE_BY_ZOMBIE
        elif state == "resources":
            if np.random.random() < 0.5:
                reward = FIND_RESOURCES
                next_state = states[0]
            else:
                reward = SURVIVE
        else:
            reward = SURVIVE
        return next_state, reward
    elif action == "move":
        # move to a new location and update the state
        if state == "zombie_nearby":
            if np.random.random() < 0.8:
                next_state = states[0]
                reward = SURVIVE
            else:
                next_state = state
                reward = DIE_BY_ZOMBIE
        else:
            next_state = np.random.choice(states[:3])
            reward = SURVIVE
        return next_state, reward
    elif action == "fight":
        # fight a nearby zombie and update the state and reward
        if state == "zombie_nearby":
            if np.random.random() < 0.9:
                reward = KILL_ZOMBIE
                if np.random.random() < 0.5:
                    next_state = state
                else:
                    next_state = states[2]
            else:
                next_state = state
                reward = DIE_BY_ZOMBIE
        else:
            next_state = state
            reward = SURVIVE
        return next_state, reward

# select an action using the learned epsilon-greedy policy
def execute_epsilon_greedy_policy(Q, state, epsilon):
    if np.random.random() < epsilon:
        # choose a random action
        action = np.random.choice(actions)
    else:
        # choose the action with the highest Q-value
        state_index = states.index(state)
        action_index = np.argmax(Q[state_index, :])
        action = actions[action_index]
    return action

# Print the Q-table
def print_q_table():
    print("Q-table:")
    for i in range(num_states):
        print(f"State: {states[i]}")
        for j in range(num_actions):
            print(f"Action: {actions[j]} - Q-value: {Q[i][j]}")
        print()
        
# define the training function
def train(num_episodes):
    # Loop through episodes
    for _ in range(num_episodes):
        # Set initial state
        current_state = np.random.choice(states)

        # Loop through steps
        while True:
            # Select an action using the epsilon-greedy policy
            epsilon = 0.25
            action = execute_epsilon_greedy_policy(Q, current_state, epsilon)

            # Execute the action and observe the resulting state and reward
            next_state, reward = execute_action(current_state, action)

            print(f"current_state: {current_state}, action: {action}, reward: {reward}")

            # Update the Q-table using the reinforcement learning algorithm
            q_learning(Q, current_state, action, reward, next_state)

            # Set the current state to the next state
            current_state = next_state

            # Check if the episode should end
            if end_condition(reward):
                break
    print_q_table()
    
# simulate the person's behavior using the learned policy
def simulate(Q):
    # Set initial state
    state = np.random.choice(states)

    # Loop through steps
    while True:
        # Select an action using the epsilon-greedy policy
        epsilon = 0.0
        action = execute_epsilon_greedy_policy(Q, state, epsilon)

        # Execute the action and observe the resulting state and reward
        next_state, reward = execute_action(state, action)

        # print the state, action, and reward
        print(f"state: {state}, action: {action}, reward: {reward}")

        # Check if the simulation should end
        if end_condition(reward):
            break
                
        # Set the current state to the next state
        state = next_state

        """
        for _ in range(num_steps):
        simulate_learned_policy(100, Q)
        """
# Train the Q-learning algorithm
train(num_episodes=1000)

# Simulate the person's behavior
# simulate(Q)

"""
The code is implementing a reinforcement learning algorithm to teach a person to survive in a zombie apocalypse. It is using a Q-learning algorithm to determine the best actions to take in different states (such as "safe_house", "zombie_nearby", or "resources"). The person can either stay, move, or fight in each state. The rewards for these actions are defined as KILL_ZOMBIE, DIE_BY_ZOMBIE, SURVIVE, and FIND_RESOURCES. The Q-table is initialized with all zeros and is updated using the Q-learning formula, which takes into account the current reward and the expected maximum reward for the next state. The person's actions are selected using an epsilon-greedy policy, which sometimes chooses a random action and other times chooses the action with the highest Q-value. The training function loops through a specified number of episodes and performs the Q-learning update and action selection for each step until the end condition is reached (the person is killed by a zombie).
"""

def try_effect_of_discount_rate():
    # define the discount rate
    discount_rates = [0.1, 0.5, 0.9, 0.99]

    # define the reward values
    rewards = list(range(100))

    # define the number of steps
    num_steps = len(rewards)

    # loop through the discount rates
    for discount_rate in discount_rates:
        # calculate the discounted rewards
        discounted_rewards = []
        for i in range(num_steps):
            discounted_rewards.append(
                rewards[i] * (discount_rate ** i))

        # plot the discounted rewards
        plt.plot(discounted_rewards, label=f"discount_rate: {discount_rate}")

    # add labels and legend
    plt.xlabel("step")
    plt.ylabel("discounted reward")
    plt.legend()
    plt.show()
    
def try_effect_of_learning_rate():
    # define the learning rates
    learning_rates = [0.1, 0.5, 0.9, 0.99]

    # define the reward values
    rewards = list(range(100))

    # define the number of steps
    num_steps = len(rewards)

    # loop through the learning rates
    for learning_rate in learning_rates:
        # calculate the discounted rewards
        discounted_rewards = []
        for i in range(num_steps):
            discounted_rewards.append(
                rewards[i] * (learning_rate ** i))

        # plot the discounted rewards
        plt.plot(discounted_rewards, label=f"learning_rate: {learning_rate}")

    # add labels and legend
    plt.xlabel("step")
    plt.ylabel("discounted reward")
    plt.legend()
    plt.show()
    
def try_effect_of_learning_rate_discount_rate():
    # define the learning rates
    learning_rates = [0.1, 0.5, 0.9, 0.99]

    # define the discount rates
    discount_rates = [0.1, 0.5, 0.9, 0.99]

    # define the reward values
    rewards = list(range(100))

    # define the number of steps
    num_steps = len(rewards)

    # loop through the learning rates
    for learning_rate in learning_rates:
        # loop through the discount rates
        for discount_rate in discount_rates:
            # calculate the discounted rewards
            discounted_rewards = []
            for i in range(num_steps):
                discounted_rewards.append(
                    rewards[i] * (learning_rate ** i) * (discount_rate ** i))

            # plot the discounted rewards
            plt.plot(discounted_rewards, label=f"learning_rate: {learning_rate}, discount_rate: {discount_rate}")

    # add labels and legend
    plt.xlabel("step")
    plt.ylabel("discounted reward")
    plt.legend()
    plt.show()
    
# try_effect_of_learning_rate_discount_rate()