import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.table import Table

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Abstract State and Action Representations
class State:
    def __init__(self, position):
        self.position = position

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def __repr__(self):
        return str(self.position)

class Action:
    def __init__(self, move):
        self.move = move

    def __eq__(self, other):
        return self.move == other.move

    def __hash__(self):
        return hash(self.move)

    def __repr__(self):
        return str(self.move)

# Grid Environment
class GridEnvironment:
    def __init__(self, grid_size, safe_zone, zombie_positions, start_position, zombie_penalty=-100, safe_zone_reward=100, step_cost=-1):
        self.grid_size = grid_size
        self.safe_zone = State(safe_zone)
        self.zombie_positions = set(State(pos) for pos in zombie_positions)
        self.start_position = State(start_position)
        self.zombie_penalty = zombie_penalty
        self.safe_zone_reward = safe_zone_reward
        self.step_cost = step_cost
        # Define actions
        self.actions = [Action((0, -1)), Action((0, 1)), Action((-1, 0)), Action((1, 0))]

    def is_terminal_state(self, state):
        return state in self.zombie_positions or state == self.safe_zone

    def get_next_state(self, state, action):
        if self.is_terminal_state(state):
            return state  # No movement in terminal state
        next_position = tuple(np.clip(np.array(state.position) + np.array(action.move), 0, self.grid_size - 1))
        return State(next_position)

    def calculate_reward(self, state):
        if state in self.zombie_positions:
            return self.zombie_penalty
        elif state == self.safe_zone:
            return self.safe_zone_reward
        return self.step_cost

    def is_valid_state(self, state):
        return 0 <= state.position[0] < self.grid_size and 0 <= state.position[1] < self.grid_size



# Temporal-Difference Learning (TD) - Q-Learning
def q_learning(env, episodes, learning_rate, discount_factor, initial_epsilon, min_epsilon, epsilon_decay, policy_store_interval=10, simulation_interval=100):
    def state_action_hash(state, action):
        return (state.position, action.move)

    def choose_action(state, q_table, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(env.actions)  # Explore
        else:
            # Exploit
            return max(env.actions, key=lambda a: q_table[state_action_hash(state, a)])

    def update_q_table(state, action, reward, next_state, q_table):
        sa_hash = state_action_hash(state, action)
        next_max = max(q_table[state_action_hash(next_state, a)] for a in env.actions)
        q_table[sa_hash] = (1 - learning_rate) * q_table[sa_hash] + learning_rate * (reward + discount_factor * next_max)
        # Set values for terminal states
        if next_state in env.zombie_positions:
            for a in env.actions:
                q_table[state_action_hash(next_state, a)] = env.zombie_penalty
        elif next_state == env.safe_zone:
            for a in env.actions:
                q_table[state_action_hash(next_state, a)] = env.safe_zone_reward

    def convert_to_policy(q_table, env):
        policy_actions = np.empty((env.grid_size, env.grid_size), dtype=object)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = State((i, j))
                best_action = max(env.actions, key=lambda a: q_table[state_action_hash(state, a)])
                policy_actions[i, j] = best_action.move
        return policy_actions

    q_table = {state_action_hash(State((i, j)), a): np.random.uniform(low=-0.1, high=0.1) for i in range(env.grid_size) for j in range(env.grid_size) for a in env.actions}
    # Set values for terminal states
    for z in env.zombie_positions:
        for a in env.actions:
            q_table[state_action_hash(z, a)] = env.zombie_penalty
    for a in env.actions:
        q_table[state_action_hash(env.safe_zone, a)] = env.safe_zone_reward
    
    epsilon = initial_epsilon
    q_learning_snapshots = []
    simulation_rewards = []

    for episode in range(episodes):
        state = env.start_position
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state = env.get_next_state(state, action)
            reward = env.calculate_reward(next_state)
            done = env.is_terminal_state(next_state)
            update_q_table(state, action, reward, next_state, q_table)

            state = next_state
            total_reward += reward
            steps += 1

        if episode % policy_store_interval == 0 or episode == episodes - 1:
            best_policy = convert_to_policy(q_table, env)
            q_learning_snapshots.append(best_policy)
            logger.info(f'Episode {episode + 1}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon:.4f}')

        if episode % simulation_interval == 0 or episode == episodes - 1:
            # Simulate episodes with the current policy
            current_policy = convert_to_policy(q_table, env)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    final_policy_q_learning = convert_to_policy(q_table, env)
    
    return final_policy_q_learning, q_learning_snapshots, simulation_rewards


# Dynamic Programming (DP) - Policy Iteration
def policy_iteration(env, episodes, gamma, theta=0.0001, policy_store_interval=1, simulation_interval=10):
    def calculate_state_value(policy, V):
        while True:
            delta = 0
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    state = State((i, j))
                    action = env.actions[policy[i, j]]
                    next_state = env.get_next_state(state, action)
                    reward = env.calculate_reward(next_state)
                    v = V[i, j]
                    V[i, j] = reward + gamma * V[next_state.position[0], next_state.position[1]]
                    delta = max(delta, abs(v - V[i, j]))
            if delta < theta:
                break

    def policy_improvement(V):
        new_policy = np.zeros((env.grid_size, env.grid_size), dtype=int)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = State((i, j))
                action_values = []
                for a in env.actions:
                    next_state = env.get_next_state(state, a)
                    reward = env.calculate_reward(next_state)
                    action_values.append(reward + gamma * V[next_state.position[0], next_state.position[1]])
                new_policy[i, j] = np.argmax(action_values)
        return new_policy

    def convert_to_policy(policy_matrix, env):
        policy_actions = np.empty((env.grid_size, env.grid_size), dtype=object)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                action_index = policy_matrix[i, j]
                action = env.actions[action_index]
                policy_actions[i, j] = action.move
        return policy_actions

    V = np.zeros((env.grid_size, env.grid_size))
    policy = np.random.randint(len(env.actions), size=(env.grid_size, env.grid_size))
    policy_snapshots = []
    simulation_rewards = []

    for episode in range(episodes):
        calculate_state_value(policy, V)
        new_policy = policy_improvement(V)

        if episode % policy_store_interval == 0 or np.array_equal(new_policy, policy):
            policy_snapshots.append(convert_to_policy(new_policy, env))
            logger.info(f'Episode {episode + 1}, Average Value: {np.mean(V):.4f}')

        if episode % simulation_interval == 0 or np.array_equal(new_policy, policy):
            # Simulate episodes with the current policy
            current_policy = convert_to_policy(policy, env)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    final_policy_dp = convert_to_policy(policy, env)

    return final_policy_dp, policy_snapshots, simulation_rewards


# Monte Carlo (MC) - First-visit MC Prediction
def monte_carlo(env, episodes, gamma, value_function_store_interval=10, simulation_interval=100):
    def convert_to_policy(env, V):
        policy = np.empty((env.grid_size, env.grid_size), dtype=tuple)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = State((i, j))
                if state in env.zombie_positions or state == env.safe_zone:
                    continue

                # Evaluate the value of each possible action from the current state
                action_values = []
                for action in env.actions:
                    next_state = env.get_next_state(state, action)
                    # Skip invalid or terminal states
                    if next_state == state or next_state in env.zombie_positions:
                        continue
                    value = V[next_state.position[0], next_state.position[1]]
                    action_values.append((action.move, value))

                # Choose the action leading to the state with highest value
                if action_values:
                    best_action = max(action_values, key=lambda x: x[1])[0]
                else:
                    raise Exception(f'No valid actions found for state {state}')

                policy[i, j] = best_action
        return policy

    V = np.zeros((env.grid_size, env.grid_size))
    # Set values for terminal states
    for z in env.zombie_positions:
        V[z.position] = env.zombie_penalty
    V[env.safe_zone.position] = env.safe_zone_reward
    V[env.start_position.position] = -np.inf
    
    returns_sum = np.zeros((env.grid_size, env.grid_size))
    returns_count = np.zeros((env.grid_size, env.grid_size))
    value_function_snapshots = []
    policy_snapshots = []
    simulation_rewards = []

    for episode in range(episodes):
        states, rewards = [], []
        state = env.start_position
        done = False

        while not done:
            action = random.choice(env.actions)
            next_state = env.get_next_state(state, action)
            states.append(state)
            reward = env.calculate_reward(next_state)
            rewards.append(reward)
            done = env.is_terminal_state(next_state)
            state = next_state

        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            if state not in states[:t] and state != env.safe_zone and state not in env.zombie_positions and state != env.start_position:
                returns_sum[state.position] += G
                returns_count[state.position] += 1
                V[state.position] = returns_sum[state.position] / returns_count[state.position]

        if episode % value_function_store_interval == 0 or episode == episodes - 1:
            value_function_snapshots.append(np.copy(V))
            current_policy = convert_to_policy(env, V)
            policy_snapshots.append(current_policy)
            logger.info(f'Episode {episode + 1}, Average Return: {np.mean(rewards):.4f}, Total States Visited: {len(set(states))}')

        if episode % simulation_interval == 0 or episode == episodes - 1:
            # Simulate episodes with the current policy
            current_policy = convert_to_policy(env, V)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

    final_policy_mc = convert_to_policy(env, V)

    return V, final_policy_mc, policy_snapshots, value_function_snapshots, simulation_rewards



def plot_policy(env, policy, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show_plot = True
    else:
        show_plot = False

    nrows, ncols = policy.shape
    ax.set_title(title)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    width, height = 1.0 / ncols, 1.0 / nrows
    action_arrows = {(-1, 0): '↑', (1, 0): '↓', (0, -1): '←', (0, 1): '→'}

    for (i, j), action in np.ndenumerate(policy):
        if (i, j) in [z.position for z in env.zombie_positions]:
            color, cell_text = 'red', 'Z'
        elif (i, j) == env.safe_zone.position:
            color, cell_text = 'green', 'S'
        elif (i, j) == env.start_position.position:
            color, cell_text = 'blue', 'Start'
        else:
            color, cell_text = 'lightgray', action_arrows.get(action, '')

        tb.add_cell(i, j, width, height, text=cell_text, loc='center', facecolor=color)

    ax.add_table(tb)
    if show_plot:
        plt.show()


def plot_value_function(env, V, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show_plot = True
    else:
        show_plot = False

    cmap = LinearSegmentedColormap.from_list('Value Function', ['white', 'yellow', 'orange', 'red'])
    im = ax.imshow(V, cmap=cmap)

    # Loop over data dimensions and create text annotations.
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            color = 'black'
            current_position = (i, j)

            if current_position in [z.position for z in env.zombie_positions]:
                text = 'Z'
            elif current_position == env.safe_zone.position:
                text = 'S'
            elif current_position == env.start_position.position:
                text = 'Start'
            else:
                text = f'{V[i, j]:.2f}'

            ax.text(j, i, text, ha="center", va="center", color=color)

    ax.set_title(title)

    if show_plot:
        fig.colorbar(im, ax=ax)
        plt.show()


def plot_policy_evolution(env, policies, title, interval=200):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        plot_policy(env, policies[frame], f"{title} - Step {frame}", ax=ax)

    ani = FuncAnimation(fig, update, frames=len(policies), interval=interval, repeat=False)
    plt.show()



def plot_value_function_evolution(env, value_functions, title, interval=200):
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = LinearSegmentedColormap.from_list('Value Function', ['white', 'yellow', 'orange', 'red'])

    im = ax.imshow(value_functions[0], cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)

    def update(frame):
        ax.clear()
        im.set_data(value_functions[frame])
        im.autoscale()
        cbar.update_normal(im)
        plot_value_function(env, value_functions[frame], f"{title} - Episode {frame}", ax=ax)

    ani = FuncAnimation(fig, update, frames=len(value_functions), interval=interval, repeat=False)
    plt.show()



def simulate_episodes(env, policy, episodes):
    total_rewards = []

    for episode in range(episodes):
        state = env.start_position
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            action_move = policy[state.position]
            action = next(a for a in env.actions if a.move == action_move)

            next_state = env.get_next_state(state, action)
            reward = env.calculate_reward(next_state)
            episode_reward += reward
            done = env.is_terminal_state(next_state)

            # Debugging output
            # print(f"Episode: {episode}, Step: {step_count}, State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")

            state = next_state
            step_count += 1

            # Safety check to prevent infinite loops in case of incorrect logic
            if step_count > 1000:  # Arbitrary large number
                print("Warning: Reached 1000 steps, terminating episode to avoid infinite loop")
                break

        total_rewards.append(episode_reward)

    # Compute and return the average reward
    avg_reward = np.mean(total_rewards)
    return avg_reward


def plot_reward_curves(reward_curves, labels):
    plt.figure(figsize=(10, 6))
    for rewards, label in zip(reward_curves, labels):
        plt.plot(rewards, label=label)
    plt.xlabel('Episodes (in intervals)')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Episodes')
    plt.legend()
    plt.show()


# Example usage
env = GridEnvironment(grid_size=5, safe_zone=(0, 0), zombie_positions=[(2, 2), (1, 3)], start_position=(4, 4))

# Q-Learning
final_policy_q_learning, q_learning_snapshots, q_learning_sim_rewards = q_learning(env, episodes=1000, learning_rate=0.1, discount_factor=0.9, initial_epsilon=0.5, min_epsilon=0.01, epsilon_decay=0.99)

# Policy Iteration
final_policy_dp, policy_snapshots, policy_iter_sim_rewards = policy_iteration(env, episodes=1000, gamma=0.9)

# Monte Carlo
V_mc, final_policy_mc, policy_mc_snapshots, V_mc_snapshots, monte_carlo_sim_rewards = monte_carlo(env, episodes=1000, gamma=0.9)

# Display reward curves
plot_reward_curves([q_learning_sim_rewards, policy_iter_sim_rewards, monte_carlo_sim_rewards], ['Q-Learning (During Training)', 'Policy Iteration (During Training)', 'Monte Carlo (During Training)'])

# Display policy evolution
plot_policy_evolution(env, q_learning_snapshots, 'Q-Learning Evolution')
plot_policy_evolution(env, policy_snapshots, 'Policy Iteration Evolution')
plot_policy_evolution(env, policy_mc_snapshots, 'Monte Carlo Evolution')
plot_value_function_evolution(env, V_mc_snapshots, 'Monte Carlo Evolution')

# Display final results
plot_policy(env, final_policy_q_learning, "Final Q-Learning Policy")
plot_policy(env, final_policy_dp, "Final Policy Iteration Policy")
plot_policy(env, final_policy_mc, "Final Monte Carlo Policy")
plot_value_function(env, V_mc, "Final Monte Carlo Value Function")

