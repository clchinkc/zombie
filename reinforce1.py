"""
Use a reinforcement learning (RL) algorithm to train the person's behavior in the zombie apocalypse.

To implement the reinforcement learning approach, we would first need to define a set of states, actions, 
and rewards that the person can experience in the zombie apocalypse. 
For example, the states could include the person's current location, the number of zombies nearby, 
and the availability of weapons or other resources. 
The actions could include moving to a new location, attacking a zombie, or hiding in a safe place. 
The rewards could reflect the person's survival, such as a positive reward for killing zombies 
and a negative reward for being killed by zombies.

Next, we would need to define a reinforcement learning algorithm, such as Q-learning, 
to learn a policy for selecting actions based on the current state and the observed rewards. 
This could be done by training the algorithm on a set of simulated scenarios, 
where the person's actions and the resulting rewards are recorded and used to update the policy.

Once the policy has been trained, we can use it to determine the person's actions in the simulation. 
This would involve observing the person's current state and selecting the action that is predicted 
to maximize the expected rewards according to the learned policy.

In this the code, we simulate the person's behavior for a fixed number of steps (1000 in this case) 
and choose a random event at each step. 
This allows the agent to learn from a variety of different scenarios and improve its policy for selecting actions. 
At the end of the simulation, we print the final q_values table to see how the agent has learned to select actions in each state.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# define the possible states as a list
states = ["hiding", "searching", "fighting"]
HIDING = states[0]
SEARCHING = states[1]
FIGHTING = states[2]

# define the possible actions as a list
actions = ["hide", "search", "fight"]
HIDE = actions[0]
SEARCH = actions[1]
FIGHT = actions[2]

# define the possible events as a list
events = ["noise", "zombie", "weapon"]
NOISE = events[0]
ZOMBIE = events[1]
WEAPON = events[2]

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
def update_q_values(Q, state, action, reward, next_state):
    # calculate the maximum expected reward for the next state
    state_index = states.index(state)
    action_index = actions.index(action)
    next_state_index = states.index(next_state)
    next_max_reward = max(Q[next_state_index][:])

    # update the Q-value for the current state and action
    Q[state_index][action_index] = (1 - LEARNING_RATE) * Q[state_index][action_index] + \
        LEARNING_RATE * (reward + DISCOUNT_RATE * next_max_reward)

# define the function for checking the end condition
def end_condition(reward):
    # end the simulation if the person is killed by a zombie
    if reward == DIE_BY_ZOMBIE:
        return True
    else:
        return False

# define the state transition function
def update_state(state, event):
    if state == HIDING:
        if event == NOISE:
            return SEARCHING
        elif event == ZOMBIE:
            return FIGHTING
        else:
            return HIDING
    elif state == SEARCHING:
        if event == ZOMBIE:
            return FIGHTING
        elif event == WEAPON:
            return SEARCHING
        else:
            return HIDING
    elif state == FIGHTING:
        if event == ZOMBIE:
            return FIGHTING
        elif event == WEAPON:
            return SEARCHING
        else:
            return HIDING

# define the action execution function
def produce_event(state, action):
    if state == HIDING:
        if action == HIDE:
            event = np.random.choice(events)
            reward = SURVIVE
        elif action == SEARCH:
            if np.random.random() < 0.5:
                event = NOISE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = SURVIVE
        elif action == FIGHT:
            event = ZOMBIE
            reward = SURVIVE
        else:
            raise ValueError("Invalid action")
    elif state == SEARCHING:
        if action == HIDE:
            event = np.random.choice(events)
            reward = SURVIVE
        elif action == SEARCH:
            if np.random.random() < 0.5:
                event = WEAPON
                reward = FIND_RESOURCES
            else:
                event = NOISE
                reward = SURVIVE
        elif action == FIGHT:
            event = np.random.choice(events)
            reward = SURVIVE
        else:
            raise ValueError("Invalid action")
    elif state == FIGHTING:
        if action == HIDE:
            if np.random.random() < 0.5:
                event = NOISE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        elif action == SEARCH:
            if np.random.random() < 0.3:
                event = WEAPON
                reward = SURVIVE
            elif np.random.random() < 0.5:
                event = ZOMBIE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        elif action == FIGHT:
            if np.random.random() < 0.9:
                if np.random.random() < 0.5:
                    event = WEAPON
                else:
                    event = ZOMBIE
                reward = KILL_ZOMBIE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        else:
            raise ValueError("Invalid action")
    else:
        raise ValueError("Invalid state")
    return event, reward

"""
# execute action function with match case statement
def execute_action(state, action):
    match state, action:
        case HIDING, HIDE:
            event = np.random.choice(events)
            reward = SURVIVE
        case HIDING, SEARCH:
            if np.random.random() < 0.5:
                event = NOISE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = SURVIVE
        case HIDING, FIGHT:
            event = ZOMBIE
            reward = SURVIVE
        case SEARCHING, HIDE:
            event = np.random.choice(events)
            reward = SURVIVE
        case SEARCHING, SEARCH:
            if np.random.random() < 0.5:
                event = WEAPON
                reward = FIND_RESOURCES
            else:
                event = NOISE
                reward = SURVIVE
        case SEARCHING, FIGHT:
            event = np.random.choice(events)
            reward = SURVIVE
        case FIGHTING, HIDE:
            if np.random.random() < 0.5:
                event = NOISE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        case FIGHTING, SEARCH:
            if np.random.random() < 0.3:
                event = WEAPON
                reward = SURVIVE
            elif np.random.random() < 0.5:
                event = ZOMBIE
                reward = SURVIVE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        case FIGHTING, FIGHT:
            if np.random.random() < 0.9:
                if np.random.random() < 0.5:
                    event = WEAPON
                else:
                    event = ZOMBIE
                reward = KILL_ZOMBIE
            else:
                event = ZOMBIE
                reward = DIE_BY_ZOMBIE
        case _, _:
            raise ValueError("Invalid state or action")
"""

# choose an action using the learned epsilon-greedy policy
def select_action(Q, state, epsilon):
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
    for i in range(num_episodes):
        # Set initial state
        current_state = np.random.choice(states)

        # Loop through steps
        while True:
            # Select an action using the epsilon-greedy policy
            epsilon = 0.25
            action = select_action(Q, current_state, epsilon)

            # Execute the action and observe the resulting state and reward
            event, reward = produce_event(current_state, action)
            
            next_state = update_state(current_state, event)

            print(f"episode: {i}, current state: {current_state}, action: {action}, event: {event}, reward: {reward}, next_state: {next_state}")

            # Update the Q-table using the reinforcement learning algorithm
            update_q_values(Q, current_state, action, reward, next_state)

            # Set the current state to the next state
            current_state = next_state

            # Check if the episode should end
            if end_condition(reward):
                break
    print_q_table()
    
# simulate the person's behavior using the learned policy
def simulate(Q):
    # Set initial state
    current_state = np.random.choice(states)

    # Loop through steps
    while True:
        # Select an action using the epsilon-greedy policy
        epsilon = 0.0
        action = select_action(Q, current_state, epsilon)

        # Execute the action and observe the resulting state and reward
        event, reward = produce_event(current_state, action)
            
        next_state = update_state(current_state, event)

        print(f"current state: {current_state}, action: {action}, event: {event}, reward: {reward}, next_state: {next_state}")

        # Check if the simulation should end
        if end_condition(reward):
            break
                
        # Set the current state to the next state
        current_state = next_state

        """
        for _ in range(num_steps):
        simulate_learned_policy(100, Q)
        """
# Train the Q-learning algorithm
train(num_episodes=1000)

# Simulate the person's behavior
# simulate(Q)

"""
The code is implementing a reinforcement learning algorithm to teach a person to survive in a zombie apocalypse. It is using a Q-learning algorithm to determine the best actions to take in different states (such as "safe_house", "fighting", or "resources"). The person can either stay, move, or fight in each state. The rewards for these actions are defined as KILL_ZOMBIE, DIE_BY_ZOMBIE, SURVIVE, and FIND_RESOURCES. The Q-table is initialized with all zeros and is updated using the Q-learning formula, which takes into account the current reward and the expected maximum reward for the next state. The person's actions are selected using an epsilon-greedy policy, which sometimes chooses a random action and other times chooses the action with the highest Q-value. The training function loops through a specified number of episodes and performs the Q-learning update and action selection for each step until the end condition is reached (the person is killed by a zombie).
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