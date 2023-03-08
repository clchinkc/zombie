# monte carlo

"""
Monte Carlo simulation can be used in a simulation of a zombie apocalypse in a number of ways. For example:
Modeling the spread of the zombie virus: Monte Carlo simulation can be used to model the random movement and interactions of people in a population, including the spread of the zombie virus from infected individuals to healthy ones.
Modeling the decision-making of survivors: Monte Carlo simulation can also be used to model the decision-making of survivors as they try to evade zombies and find safety. For example, it can simulate the probability of a survivor successfully reaching a safe haven given various factors such as their starting location, the number of zombies in the area, and their supplies.
Modeling the impact of interventions: Monte Carlo simulation can be used to model the impact of various interventions on the spread of the zombie virus and the survival of the human population. For example, it can simulate the effect of a quarantine on the number of new zombie infections or the impact of a vaccine on the number of survivors.
Generating scenarios: Monte Carlo simulation can be used to generate a range of possible scenarios for a zombie apocalypse, taking into account the uncertainty and randomness inherent in such a situation. This can provide valuable insights into the range of possible outcomes and help decision-makers prepare for a wide range of contingencies.
In all these cases, Monte Carlo simulation can provide valuable insights into the dynamics of a zombie apocalypse and help decision-makers prepare for such a scenario. However, it's important to note that the results of a Monte Carlo simulation are only as good as the model used, and it's crucial to carefully validate the assumptions and inputs used in the simulation.
"""
"""
Here's an example of how you could design a Monte Carlo simulation of a zombie apocalypse:
Define the population and their initial state: Start by defining the total population and their initial state, such as their location, health status (healthy or infected), and other relevant attributes. You could represent each individual in the population as an object in your simulation with properties such as location, health status, and decision-making capabilities.
Model the spread of the zombie virus: Next, you'll need to model the spread of the zombie virus from infected individuals to healthy ones. This could be based on factors such as the proximity of healthy individuals to infected ones, the number of zombies in the area, and the likelihood of a healthy individual being bitten or coming into contact with infected fluids.
Model the decision-making of survivors: You'll also need to model the decision-making of survivors as they try to evade zombies and find safety. This could be based on factors such as the proximity of safe havens, the number of zombies in the area, and the supplies they have on hand. You could use random numbers to model the uncertainty and randomness inherent in these decisions.
Simulate the movement of the population: Next, you'll simulate the movement of the population over time. This could involve updating the location of each individual in the simulation based on their decisions and the factors affecting their movement, such as the presence of zombies and safe havens.
Keep track of the state of the population: Keep track of the state of the population, including the number of healthy individuals, the number of infected individuals, and the number of fatalities. You could also keep track of other relevant metrics, such as the number of safe havens, the amount of supplies, and the distribution of the population over the landscape.
Run the simulation multiple times: Finally, you'll want to run the simulation multiple times, perhaps hundreds or thousands of times, to generate a range of possible scenarios. This will provide valuable insights into the range of possible outcomes and help you prepare for a wide range of contingencies.
Analyze the results: Once you have run the simulation multiple times, you can analyze the results to get a better understanding of the dynamics of the zombie apocalypse. For example, you could calculate statistics such as the average number of fatalities, the average number of survivors, and the average time to reach a safe haven. You could also generate graphs and visualizations to help you understand the results and identify any patterns or trends.
This is just one example of how you could design a Monte Carlo simulation of a zombie apocalypse. Of course, the specific details will depend on the goals of your simulation and the assumptions you make about the underlying processes. However, by using a Monte Carlo simulation, you can gain valuable insights into the dynamics of a zombie apocalypse and help prepare for such a scenario.
"""
"""
Here's a more specific example of how you could use Monte Carlo simulation to model a zombie apocalypse:
Define the input variables: Start by defining the input variables that you want to include in your simulation, such as the initial number of healthy individuals, the initial number of infected individuals, the rate of spread of the zombie virus, and the likelihood of a healthy individual becoming infected after being bitten.
Generate random numbers: Use random number generators to simulate the uncertainty inherent in the spread of the zombie virus and the decisions made by survivors. For example, you could generate random numbers to determine the number of bites that a healthy individual receives and the time it takes for them to become infected.
Run the simulation: Using the input variables and random numbers, run the simulation over a specified number of time steps. Keep track of the number of healthy individuals, the number of infected individuals, and the number of fatalities at each time step.
Repeat the simulation: Repeat the simulation multiple times, each time using different random numbers. This will give you a range of possible outcomes and help you understand the distribution of outcomes for various scenarios.
Analyze the results: After running the simulation multiple times, analyze the results to get a better understanding of the dynamics of the zombie apocalypse. For example, you could calculate the average number of fatalities, the average number of survivors, and the time it takes for the zombie virus to spread through the population. You could also create visualizations to help you understand the results and identify any patterns or trends.
This is a simple example of how you could use Monte Carlo simulation to model a zombie apocalypse. In practice, you could include additional variables and make more complex models to better reflect the underlying processes. The key idea is to use random numbers and simulation to model the uncertainty and randomness inherent in the spread of the zombie virus and the decisions made by survivors. By using Monte Carlo simulation, you can gain valuable insights into the dynamics of a zombie apocalypse and help prepare for such a scenario.
"""

# without: Markov chain Monte Carlo, Markov Decision Processes, https://pojenlai.wordpress.com/2018/11/14/paper-reading-the-naive-utility-calculus-joint-inferences-about-the-costs-and-rewards-of-actions/

# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/montecarlo1.ipynb
# https://franklin.dyer.me/post/119


"""
import itertools
import operator
import random
from time import sleep

# Import the necessary packages and modules
import gym
import numpy as np
import tqdm

tqdm.monitor_interval = 0

# Random Policy
def create_random_policy(env):
    policy = {}
    for key in range(0, env.observation_space.n):
        current_end = 0
        p = {}
    for action in range(0, env.action_space.n):
        p[action] = 1 / env.action_space.n
        policy[key] = p
        return policy


# Create a state-action dictionary
def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
        return Q


# Create run_game function
def run_game(env, policy, display=True):
    env.reset()
    episode = []
    finished = False
    while not finished:
        s = env.env.s
        if display:
            clear_output(True)
            env.render()
            sleep(1)
            timestep = []
            timestep.append(s)
            n = random.uniform(0, sum(policy[s].values()))
            top_range = 0
    for prob in policy[s].items():
        top_range += prob[1]
        if n < top_range:
            action = prob[0]
        break
    state, reward, finished, info = env.step(action)
    timestep.append(action)
    timestep.append(reward)
    episode.append(timestep)
    if display:
        clear_output(True)
        env.render()
        sleep(1)
    return episode


# Create test_policy function
def test_policy(policy, env):
    wins = 0
    r = 100
    for i in range(r):
        w = run_game(env, policy, display=False)[-1][-1]
    if w == 1:
        wins += 1
    return wins / r


import sys

epsilon = sys.float_info.epsilon
epsilon = 0.1

# Create Monte Carlo e-soft function
def monte_carlo_e_soft(env, episodes=100, policy=None, epsilon=0.01):
    if not policy:
        policy = create_random_policy(
            env
        )  # Create an empty dictionary to store state action values
    Q = create_state_action_dictionary(
        env, policy
    )  # Empty dictionary for storing rewards for each state-action pair
    returns = {}  # 3.
    for _ in range(episodes):  # Looping through episodes
        G = 0  # Store cumulative reward in G (initialized at 0)
        episode = run_game(
            env=env, policy=policy, display=False
        )  # Store state, action and value respectively
    # for loop through reversed indices of episode array.
    # The logic behind it being reversed is that the eventual reward would be at the end.
    # So we have to go back from the last timestep to the first one propagating result from the future.
    for i in reversed(range(0, len(episode))):
        s_t, a_t, r_t = episode[i]
        state_action = (s_t, a_t)
        G += r_t  # Increment total reward by reward on current timestep
    if not state_action in [(x[0], x[1]) for x in episode[0:i]]:  #
        if returns.get(state_action):
            returns[state_action].append(G)
        else:
            returns[state_action] = [G]
        Q[s_t][a_t] = sum(returns[state_action]) / len(
            returns[state_action]
        )  # Average reward across episodes
        Q_list = list(
            map(lambda x: x[1], Q[s_t].items())
        )  # Finding the action with maximum value
        indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
        max_Q = random.choice(indices)
        A_star = max_Q  # 14.
        for a in policy[s_t].items():  # Update action probability for s_t in policy
            if a[0] == A_star:
                policy[s_t][a[0]] = (
                    1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                )
            else:
                policy[s_t][a[0]] = epsilon / abs(sum(policy[s_t].values()))


env = gym.make("FrozenLake8x8-v0")
policy = monte_carlo_e_soft(env, episodes=5)
test_policy(policy, env)
"""

