import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.table import Table
from matplotlib.transforms import Bbox


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
        self.zombie_positions = list(map(State, zombie_positions))
        self.start_position = State(start_position)
        self.zombie_penalty = zombie_penalty
        self.safe_zone_reward = safe_zone_reward
        self.step_cost = step_cost
        # Define rewards
        self.rewards = {State(pos): step_cost for pos in np.ndindex(grid_size, grid_size)}
        self.rewards[State(safe_zone)] = self.safe_zone_reward
        for z in self.zombie_positions:
            self.rewards[z] = self.zombie_penalty
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
        return self.rewards.get(state, self.step_cost)

    def is_valid_state(self, state):
        return 0 <= state.position[0] < self.grid_size and 0 <= state.position[1] < self.grid_size

    def step(self, state, action):
        next_state = self.get_next_state(state, action)
        reward = self.calculate_reward(next_state)
        done = self.is_terminal_state(next_state)
        return next_state, reward, done

    def reset(self):
        return self.start_position


# Q-Learning
def q_learning(env, logger, episodes, initial_learning_rate, min_learning_rate, learning_rate_decay, discount_factor, initial_epsilon, min_epsilon, epsilon_decay, policy_store_interval=10, simulation_interval=100):
    def state_action_index(state, action):
        return state.position[0], state.position[1], env.actions.index(action)

    def choose_action(state, q_table, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(env.actions)  # Explore
        else:
            # Exploit
            state_index = state.position[0], state.position[1]
            return env.actions[np.argmax(q_table[state_index])]

    def update_q_table(state, action, reward, next_state, q_table):
        current_index = state_action_index(state, action)
        next_state_index = next_state.position[0], next_state.position[1]
        next_max = np.max(q_table[next_state_index])
        q_table[current_index] = (1 - learning_rate) * q_table[current_index] + learning_rate * (reward + discount_factor * next_max)

    def convert_to_policy(q_table, env):
        policy_actions = np.empty((env.grid_size, env.grid_size), dtype=object)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state_index = (i, j)
                best_action_index = np.argmax(q_table[state_index])
                policy_actions[i, j] = env.actions[best_action_index].move
        return policy_actions

    num_actions = len(env.actions)
    q_table = np.random.uniform(low=-0.1, high=0.1, size=(env.grid_size, env.grid_size, num_actions))

    # Set values for terminal states
    for z in env.zombie_positions:
        for a in env.actions:
            q_table[state_action_index(z, a)] = env.zombie_penalty
    for a in env.actions:
        q_table[state_action_index(env.safe_zone, a)] = env.safe_zone_reward
    
    epsilon = initial_epsilon
    learning_rate = initial_learning_rate
    q_learning_snapshots = []
    simulation_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = env.step(state, action)
            update_q_table(state, action, reward, next_state, q_table)

            state = next_state
            total_reward += reward
            steps += 1

        if episode % policy_store_interval == 0 or episode == episodes - 1:
            best_policy = convert_to_policy(q_table, env)
            q_learning_snapshots.append(best_policy)
            logger.info("Episode completed", episode=episode+1, total_reward=total_reward, steps=steps, epsilon=epsilon, learning_rate=learning_rate)

        if episode % simulation_interval == 0 or episode == episodes - 1:
            # Simulate episodes with the current policy
            current_policy = convert_to_policy(q_table, env)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        learning_rate = max(min_learning_rate, learning_rate * learning_rate_decay)
    
    final_policy_q_learning = convert_to_policy(q_table, env)
    
    return final_policy_q_learning, q_learning_snapshots, simulation_rewards

# TD-Lambda
def td_lambda(env, logger, episodes, gamma, lambda_, initial_learning_rate, min_learning_rate, learning_rate_decay, initial_epsilon, min_epsilon, epsilon_decay, policy_store_interval=10, simulation_interval=100):
    def state_action_index(state, action):
        return state.position[0], state.position[1], env.actions.index(action)

    def choose_action(state, q_table, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(env.actions)  # Explore
        else:
            state_index = state.position[0], state.position[1]
            return env.actions[np.argmax(q_table[state_index])]  # Exploit

    def convert_to_policy(q_table, env):
        policy_actions = np.empty((env.grid_size, env.grid_size), dtype=object)
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state_index = (i, j)
                best_action_index = np.argmax(q_table[state_index])
                policy_actions[i, j] = env.actions[best_action_index].move
        return policy_actions

    num_actions = len(env.actions)
    q_table = np.random.uniform(low=-0.1, high=0.1, size=(env.grid_size, env.grid_size, num_actions))
    eligibility_traces = np.zeros_like(q_table)
    td_lambda_snapshots = []
    simulation_rewards = []

    epsilon = initial_epsilon
    learning_rate = initial_learning_rate

    for episode in range(episodes):
        state = env.reset()
        action = choose_action(state, q_table, epsilon)
        done = False
        eligibility_traces.fill(0)  # Reset eligibility traces at the start of each episode

        while not done:
            next_state, reward, done = env.step(state, action)
            next_action = choose_action(next_state, q_table, epsilon)  # Choose next action based on the policy

            td_error = reward + gamma * q_table[state_action_index(next_state, next_action)] - q_table[state_action_index(state, action)]
            eligibility_traces[state_action_index(state, action)] += 1

            # Update Q-values and traces for all state-action pairs
            for s in np.ndindex(env.grid_size, env.grid_size):
                for a in range(num_actions):
                    sa_index = (s[0], s[1], a)
                    q_table[sa_index] += learning_rate * td_error * eligibility_traces[sa_index]
                    eligibility_traces[sa_index] *= gamma * lambda_

            state, action = next_state, next_action

        # Adjust epsilon and learning rate after each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        learning_rate = max(min_learning_rate, learning_rate * learning_rate_decay)

        # Store policy and log information at intervals
        if episode % policy_store_interval == 0 or episode == episodes - 1:
            best_policy = convert_to_policy(q_table, env)
            td_lambda_snapshots.append(best_policy)
            logger.info("Episode completed", episode=episode+1, td_error=td_error, epsilon=epsilon, learning_rate=learning_rate)

        # Simulate with current policy at intervals
        if episode % simulation_interval == 0 or episode == episodes - 1:
            current_policy = convert_to_policy(q_table, env)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

    final_policy_td_lambda = convert_to_policy(q_table, env)
    return final_policy_td_lambda, td_lambda_snapshots, simulation_rewards



# Value Iteration
def value_iteration(env, logger, episodes, gamma=0.9, theta=1e-3, snapshot_interval=100, simulation_interval=100):
    def one_step_lookahead(state, V):
        action_values = np.zeros(len(env.actions))
        for a, action in enumerate(env.actions):
            next_state = env.get_next_state(state, action)
            reward = env.calculate_reward(next_state)
            action_values[a] = reward + gamma * V[next_state.position]
        return action_values

    V = np.zeros((env.grid_size, env.grid_size))
    policy_value_snapshots = []
    value_function_snapshots = []
    simulation_rewards = []

    for episode in range(episodes):
        delta = 0
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = State((i, j))
                if env.is_terminal_state(state):
                    continue

                A = one_step_lookahead(state, V)
                best_action_value = np.max(A)
                delta = max(delta, np.abs(best_action_value - V[i, j]))
                V[i, j] = best_action_value

        if delta < theta:
            break

        if episode % snapshot_interval == 0:
            # Snapshot of policy and value function
            current_policy = np.zeros((env.grid_size, env.grid_size), dtype=object)
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    state = State((i, j))
                    if env.is_terminal_state(state):
                        continue

                    A = one_step_lookahead(state, V)
                    best_action = np.argmax(A)
                    current_policy[i, j] = env.actions[best_action].move

            policy_value_snapshots.append(np.copy(current_policy))
            value_function_snapshots.append(np.copy(V))
            logger.info("Episode completed", episode=episode+1, average_value=np.mean(V))

            if episode % simulation_interval == 0:
                # Simulate episodes with the current policy
                rewards = simulate_episodes(env, current_policy, episodes=10)
                simulation_rewards.append(rewards)

    # Final policy
    final_policy_value = np.zeros((env.grid_size, env.grid_size), dtype=object)
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            state = State((i, j))
            if env.is_terminal_state(state):
                continue

            A = one_step_lookahead(state, V)
            best_action = np.argmax(A)
            final_policy_value[i, j] = env.actions[best_action].move

    return V, final_policy_value, policy_value_snapshots, value_function_snapshots, simulation_rewards

# Policy Iteration
def policy_iteration(env, logger, episodes, gamma, theta=0.0001, policy_store_interval=1, simulation_interval=10):
    def calculate_state_value(policy, V):
        while True:
            new_V = np.zeros_like(V)
            for action_index, action in enumerate(env.actions):
                # Calculate next states for each action
                action_move = np.array(action.move)
                next_positions = np.clip(np.dstack(np.indices(V.shape)) + action_move, 0, env.grid_size - 1)
                next_states = next_positions.reshape(-1, 2)

                # Vectorized check for valid and non-terminal states
                valid_mask = np.all((next_states >= 0) & (next_states < env.grid_size), axis=1)
                terminal_mask = np.array([env.is_terminal_state(State(tuple(pos))) for pos in next_states])
                combined_mask = valid_mask & ~terminal_mask
                combined_mask = combined_mask.reshape(V.shape)

                # Apply policy and combined mask
                policy_combined_mask = (policy == action_index) & combined_mask

                # Vectorized calculation of rewards
                rewards = np.array([env.calculate_reward(State(tuple(pos))) for pos in next_states])
                rewards = rewards.reshape(V.shape)

                # Update value function
                next_values = V[next_positions[..., 0], next_positions[..., 1]]
                new_V[policy_combined_mask] = rewards[policy_combined_mask] + gamma * next_values[policy_combined_mask]

            delta = np.max(np.abs(new_V - V))
            V[:] = new_V

            if delta < theta:
                break

    def policy_improvement(V):
        # Initialize new policy
        new_policy = np.zeros((env.grid_size, env.grid_size), dtype=int)

        # Calculate next states for each action
        all_actions = np.array([a.move for a in env.actions])  # Shape: [num_actions, 2]
        grid_indices = np.dstack(np.indices(V.shape))  # Shape: [grid_size, grid_size, 2]
        next_states = grid_indices[:, :, None, :] + all_actions[None, None, :, :]  # Shape: [grid_size, grid_size, num_actions, 2]

        # Clip next states to valid grid positions
        valid_next_states = np.clip(next_states, 0, env.grid_size - 1)

        # Calculate rewards and values for next states
        next_state_rewards = np.array([[env.calculate_reward(State(tuple(pos))) for pos in valid_next_states[i, j]] for i in range(env.grid_size) for j in range(env.grid_size)]).reshape(env.grid_size, env.grid_size, len(env.actions))
        next_state_values = V[valid_next_states[..., 0], valid_next_states[..., 1]]  # Shape: [grid_size, grid_size, num_actions]

        # Compute action values
        action_values = next_state_rewards + gamma * next_state_values  # Shape: [grid_size, grid_size, num_actions]

        # Update policy
        new_policy = np.argmax(action_values, axis=2)

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
            logger.info("Episode completed", episode=episode+1, average_value=np.mean(V))

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

# Monte Carlo
def monte_carlo(env, logger, episodes, gamma, value_function_store_interval=10, simulation_interval=100):
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
            next_state, reward, done = env.step(state, action)
            states.append(state)
            rewards.append(reward)
            state = next_state

        unique_states = list(set(states))
        G = np.zeros((env.grid_size, env.grid_size))
        for state in unique_states:
            first_occurrence_idx = next(i for i, x in enumerate(states) if x == state)
            G[state.position] = sum([gamma ** i * r for i, r in enumerate(rewards[first_occurrence_idx:])])

        for state in unique_states:
            if state not in env.zombie_positions and state != env.safe_zone and state != env.start_position:
                returns_sum[state.position] += G[state.position]
                returns_count[state.position] += 1
                V[state.position] = returns_sum[state.position] / returns_count[state.position]

        if episode % value_function_store_interval == 0 or episode == episodes - 1:
            value_function_snapshots.append(np.copy(V))
            current_policy = convert_to_policy(env, V)
            policy_snapshots.append(current_policy)
            logger.info("Episode completed", episode=episode+1, average_return=np.mean(rewards), average_value=np.nanmean(np.where(np.ravel_multi_index(env.start_position.position, V.shape) != np.arange(V.size).reshape(V.shape), V, np.nan)))

        if episode % simulation_interval == 0 or episode == episodes - 1:
            # Simulate episodes with the current policy
            current_policy = convert_to_policy(env, V)
            rewards = simulate_episodes(env, current_policy, episodes=10)
            simulation_rewards.append(rewards)

    final_policy_mc = convert_to_policy(env, V)

    return V, final_policy_mc, policy_snapshots, value_function_snapshots, simulation_rewards


def simulate_episodes(env, policy, episodes):
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            action_move = policy[state.position]
            action = next(a for a in env.actions if a.move == action_move)

            next_state, reward, done = env.step(state, action)
            episode_reward += reward

            # Debugging output
            # print(f"Episode: {episode}, Step: {step_count}, State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")

            state = next_state
            step_count += 1

            # Safety check to prevent infinite loops in case of incorrect logic
            if step_count >= 1000:  # Arbitrary large number
                logger.info(f"Warning: Did not finish after {step_count} steps, terminating episode to avoid infinite loop")
                break

        total_rewards.append(episode_reward)

    # Compute and return the average reward
    avg_reward = np.mean(total_rewards)
    return avg_reward


class Logger:
    def __init__(self):
        # Set up the basic configuration for the logger
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('Reinforcement Learning Logger')

    def debug(self, message, **kwargs):
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message, **kwargs):
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message, **kwargs):
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message, **kwargs):
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message, **kwargs):
        self.logger.critical(self._format_message(message, kwargs))

    def _format_message(self, message, kwargs):
        context = ' '.join([f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}' for key, value in kwargs.items()])
        return f'{message} - {context}' if context else message


class Visualization:
    def __init__(self, env):
        self.env = env

    def plot_policy(self, policy, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show_plot = True
        else:
            show_plot = False

        nrows, ncols = policy.shape
        ax.set_title(title)
        ax.set_axis_off()
        bbox = Bbox.from_bounds(0, 0, 1, 1)
        tb = Table(ax, bbox=bbox)

        width, height = 1.0 / ncols, 1.0 / nrows
        action_arrows = {(-1, 0): '↑', (1, 0): '↓', (0, -1): '←', (0, 1): '→'}

        for (i, j), action in np.ndenumerate(policy):
            cell_text, color = self._get_cell_text_and_color(i, j, action, action_arrows)
            tb.add_cell(i, j, width, height, text=cell_text, loc='center', facecolor=color)

        ax.add_table(tb)
        if show_plot:
            plt.show()

    def plot_value_function(self, V, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show_plot = True
        else:
            show_plot = False

        cmap = LinearSegmentedColormap.from_list('Value Function', ['white', 'yellow', 'orange', 'red'])
        im = ax.imshow(V, cmap=cmap)

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                text, color = self._get_value_text_and_color(i, j, V)
                ax.text(j, i, text, ha="center", va="center", color=color)

        ax.set_title(title)

        if show_plot:
            fig.colorbar(im, ax=ax)
            plt.show()

    def animate_policy_evolution(self, policies, title, interval=200):
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()
            self.plot_policy(policies[frame], f"{title} - Step {frame}", ax=ax)

        ani = FuncAnimation(fig, update, frames=len(policies), interval=interval, repeat=False)
        plt.show()

    def animate_value_function_evolution(self, value_functions, title, interval=200):
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = LinearSegmentedColormap.from_list('Value Function', ['white', 'yellow', 'orange', 'red'])

        im = ax.imshow(value_functions[0], cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)

        def update(frame):
            ax.clear()
            im.set_data(value_functions[frame])
            im.autoscale()
            cbar.update_normal(im)
            self.plot_value_function(value_functions[frame], f"{title} - Episode {frame}", ax=ax)

        ani = FuncAnimation(fig, update, frames=len(value_functions), interval=interval, repeat=False)
        plt.show()

    def _get_cell_text_and_color(self, i, j, action, action_arrows):
        if (i, j) in [z.position for z in self.env.zombie_positions]:
            color, cell_text = 'red', 'Z'
        elif (i, j) == self.env.safe_zone.position:
            color, cell_text = 'green', 'S'
        elif (i, j) == self.env.start_position.position:
            color, cell_text = 'blue', 'Start'
        else:
            color, cell_text = 'lightgray', action_arrows.get(action, '')
        return cell_text, color

    def _get_value_text_and_color(self, i, j, V):
        color = 'black' if V[i, j] < V.max() / 2 else 'white'
        current_position = (i, j)

        if current_position in [z.position for z in self.env.zombie_positions]:
            text = 'Z'
        elif current_position == self.env.safe_zone.position:
            text = 'S'
        elif current_position == self.env.start_position.position:
            text = 'Start'
        else:
            text = f'{V[i, j]:.2f}'
        return text, color

    def plot_reward_curves(self, reward_curves, labels):
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
logger = Logger()
visualization = Visualization(env)

# Q-Learning
final_policy_q_learning, q_learning_snapshots, q_learning_sim_rewards = q_learning(env, logger, episodes=1000, initial_learning_rate=0.1, min_learning_rate=0.01, learning_rate_decay=0.99, discount_factor=0.9, initial_epsilon=0.5, min_epsilon=0.01, epsilon_decay=0.99)

# TD-Lambda
final_policy_td_lambda, td_lambda_snapshots, td_lambda_sim_rewards = td_lambda(env, logger, episodes=1000, gamma=0.9, lambda_=0.9, initial_learning_rate=0.1, min_learning_rate=0.01, learning_rate_decay=0.99, initial_epsilon=0.5, min_epsilon=0.01, epsilon_decay=0.999)

# Value Iteration
V_value, final_policy_value, policy_value_snapshots, V_value_snapshots, value_iter_sim_rewards = value_iteration(env, logger, episodes=1000, gamma=0.9)

# Policy Iteration
final_policy_dp, policy_snapshots, policy_iter_sim_rewards = policy_iteration(env, logger, episodes=1000, gamma=0.9)

# Monte Carlo
V_mc, final_policy_mc, policy_mc_snapshots, V_mc_snapshots, monte_carlo_sim_rewards = monte_carlo(env, logger, episodes=1000, gamma=0.9)

# Display reward curves
visualization.plot_reward_curves([q_learning_sim_rewards, td_lambda_sim_rewards, value_iter_sim_rewards, policy_iter_sim_rewards, monte_carlo_sim_rewards], ['Q-Learning', 'TD-Lambda', 'Value Iteration', 'Policy Iteration', 'Monte Carlo'])

# Display policy evolution
visualization.animate_policy_evolution(q_learning_snapshots, 'Q-Learning Evolution')
visualization.animate_policy_evolution(td_lambda_snapshots, 'TD-Lambda Evolution')
visualization.animate_policy_evolution(policy_value_snapshots, 'Value Iteration Evolution')
visualization.animate_policy_evolution(policy_snapshots, 'Policy Iteration Evolution')
visualization.animate_policy_evolution(policy_mc_snapshots, 'Monte Carlo Evolution')
visualization.animate_value_function_evolution(V_value_snapshots, 'Value Iteration Evolution')
visualization.animate_value_function_evolution(V_mc_snapshots, 'Monte Carlo Evolution')

# Display final results
visualization.plot_policy(final_policy_q_learning, "Final Q-Learning Policy")
visualization.plot_policy(final_policy_td_lambda, "Final TD-Lambda Policy")
visualization.plot_policy(final_policy_value, "Final Value Iteration Policy")
visualization.plot_policy(final_policy_dp, "Final Policy Iteration Policy")
visualization.plot_policy(final_policy_mc, "Final Monte Carlo Policy")
visualization.plot_value_function(V_value, "Final Value Iteration Value Function")
visualization.plot_value_function(V_mc, "Final Monte Carlo Value Function")

"""
**Vectorization with NumPy**:
    - You are currently using for-loops for operations on grid-like data structures. These can be optimized using NumPy's vectorized operations. For example, in functions like convert_to_policy used across different methods, you can use NumPy's array operations to directly map actions to their corresponding moves to simplify the code while optimize the performance.

**Refactoring for Code Reuse**:
    - There are some common functionalities across different methods (like `convert_to_policy`). You might want to refactor these into separate functions to avoid code duplication.

Please move all algorithms into agent class and let the agent class have a method to run the algorithm and return the policy and value function to move the agent in the grid.

State and Action Space Analysis: Add functions to analyze the state and action spaces, such as frequency of visiting each state and selecting each action. This could provide insight into the behavior of the agent and the effectiveness of the exploration strategy.

Visualization Enhancements:
Parameter Tuning Interface: Implement a user-friendly interface or a set of functions to easily modify and experiment with different hyperparameters like learning rate, discount factor, epsilon values, etc.
Consider using interactive visualizations (like using matplotlib.widgets or a web-based solution) for a more dynamic and informative experience.
Algorithm Comparisons: Develop a comparison framework to evaluate the performance of each algorithm under similar conditions. This could include metrics like convergence speed, final reward, or robustness to changes in the environment.

Environment Complexity: Introduce more complex environments with dynamic obstacles or changing rewards to test the adaptability and robustness of the algorithms.

Extend to Multi-Agent Scenarios: Expand the environment to support multiple agents, which could lead to more complex and interesting interactions and learning dynamics.
"""
"""
class Agent:
    def __init__(self, env, gamma=0.99, epsilon=1e-5, algorithm):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.algorithm = algorithm
        
        try:
            self.V, self.policy = self.algorithm(env, gamma, epsilon)
        except TypeError:
            raise TypeError("The algorithm must be a function that takes the environment, gamma, and epsilon as arguments.")
    
    def move(self, state):
        return self.env.get_next_state(state, self.policy[state])

"""
# https://www.datascienceblog.net/post/reinforcement-learning/mdps_dynamic_programming/
# Monte Carlo Policy Gradient
# actor critic
# about monte carlo
# https://blog.csdn.net/hiwallace/article/details/81284799
# Sarsa、Qlearning；蒙特卡洛策略、时序差分等
# https://cloud.tencent.com/developer/article/2338239?areaId=106001
# gym environment+pytorch
# https://zhiqingxiao.github.io/rl-book/en2023/code/CartPole-v0_VPG_torch.html
# gym environment+tensorflow
# https://zhiqingxiao.github.io/rl-book/en2023/code/CartPole-v0_VPG_tf.html

