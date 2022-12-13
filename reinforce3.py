import numpy as np

# Define states, actions, and rewards
states = ["location1", "location2", "location3", "location4"]
actions = ["move", "attack", "hide"]
rewards = {"kill_zombie": 1, "death": -1, "survive": 0}

# Initialize Q-table with all zeros
Q = np.zeros((len(states), len(actions)))

# Define Q-learning parameters
alpha = 0.1
gamma = 0.6
num_episodes = 1000

# Loop through episodes
for episode in range(num_episodes):
    # Set initial state
    current_state = np.random.choice(states)

    # Loop through steps in episode
    while current_state != "location4":
        # Choose action according to epsilon-greedy policy
        if np.random.uniform(0, 1) < 0.5:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[current_state, :])]

        # Take action and observe new state and reward
        if action == "move":
            new_state = np.random.choice(states[:3])
            reward = rewards["survive"]
        elif action == "attack":
            new_state = current_state
            if np.random.uniform(0, 1) < 0.8:
                reward = rewards["kill_zombie"]
            else:
                reward = rewards["death"]
        else:
            new_state = current_state
            reward = rewards["survive"]

        # Update Q-table
        Q[current_state, action] = Q[current_state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[current_state, action])

        # Set current state to new state
        current_state = new_state

# Print final Q-table
print(Q)

# Test learned policy
current_state = "location1"
while current_state != "location4":
    action = actions[np.argmax(Q[current_state, :])]
    print(f"Take action: {action} in state: {current_state}")
    if action == "move":
        new_state = np.random.choice(states[:3])
    elif action == "attack":
        new_state = current_state
        if np.random.uniform(0, 1) < 0.8:
            print("Killed zombie!")
        else:
            print("You died!")
            break
    else:
        new_state = current_state
    current_state = new_state

print("You survived the zombie apocalypse!")