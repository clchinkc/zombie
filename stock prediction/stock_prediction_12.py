
import random

import gym
import numpy as np
import pandas as pd
from gym import spaces


class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, window_size=5, initial_balance=1000, buy_fee=0.001, sell_fee=0.001):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size,))

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.current_step = self.window_size - 1

        return self.stock_data[self.current_step - self.window_size + 1:self.current_step + 1].values

    def step(self, action):
        prev_step = self.current_step
        self.current_step += 1
        stock_price = self.stock_data[self.current_step]

        if action == 0:  # Buy
            self.stock_owned += 1
            self.balance -= stock_price * (1 + self.buy_fee)
        elif action == 2:  # Sell
            self.stock_owned -= 1
            self.balance += stock_price * (1 - self.sell_fee)

        reward = self.balance + self.stock_owned * stock_price - self.initial_balance
        done = self.current_step == len(self.stock_data) - 1

        next_state = self.stock_data[self.current_step - self.window_size + 1:self.current_step + 1].values

        return next_state, reward, done, {}


class QLearningAgent:
    def __init__(self, n_actions, n_states, n_bins=10, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_bins = n_bins
        self.q_table = {}

    def _discretize_state(self, state):
        discrete_state = np.floor(state / self.n_bins).astype(int)
        index = tuple(discrete_state)
        return index

    def _initialize_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

    def choose_action(self, state):
        discrete_state = self._discretize_state(state)
        self._initialize_state(discrete_state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state):
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        self._initialize_state(discrete_state)
        self._initialize_state(discrete_next_state)
        predict = self.q_table[discrete_state][action]
        target = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        self.q_table[discrete_state][action] += self.alpha * (target - predict)



# Load historical stock data (ensure that it is daily closing prices)
stock_data = pd.read_csv("apple_stock_data.csv")["Close"]

# Create the custom gym environment
env = StockTradingEnv(stock_data)

# Create the Q-learning agent
agent = QLearningAgent(n_actions=3, n_states=len(stock_data) - env.window_size)

# Train the Q-learning
for episode in range(100):
    state = env.reset()
    action = agent.choose_action(state)
    done = False

    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.update(state, action, reward, next_state)

        state = next_state
        action = next_action
        
# Test the Q-learning
state = env.reset()
action = agent.choose_action(state)
done = False

while not done:
    next_state, reward, done, _ = env.step(action)
    next_action = agent.choose_action(next_state)

    state = next_state
    action = next_action
    
# Print the results
print("Balance: ", env.balance)
print("Stock owned: ", env.stock_owned)
print("Current step: ", env.current_step)
print("Stock price: ", env.stock_data[env.current_step])
print("Total value: ", env.balance + env.stock_owned * env.stock_data[env.current_step])

# Plot the results
import matplotlib.pyplot as plt

plt.plot(env.stock_data)
plt.show()
