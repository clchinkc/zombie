
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

# Input variables
num_simulations = 10
time_steps = 100
initial_healthy = 100
initial_infected = 10
infection_rate = 0.01
killing_rate = 0.005
CITY_SIZE = 100

class Person:
    def __init__(self, speed_mean, speed_std, position=None, infected=False):
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.speed = np.random.normal(self.speed_mean, self.speed_std)
        self.position = position if position is not None else np.random.uniform(0, CITY_SIZE)
        self.infected = infected

    def update_position(self):
        self.position += self.speed
        self.position = np.clip(self.position, 0, CITY_SIZE)

def catch_probability(zombie, human):
    z = (human.speed_mean - zombie.speed_mean) / np.sqrt(human.speed_std**2 + zombie.speed_std**2)
    p = 0.5 * (1 + erf(z / np.sqrt(2)))
    return p

def run_simulation():
    healthy_population = [initial_healthy]
    infected_population = [initial_infected]
    total_population = initial_healthy + initial_infected

    humans = [Person(5.0, 1.5) for _ in range(initial_healthy)]
    zombies = [Person(1.0, 1.0, infected=True) for _ in range(initial_infected)]

    for _ in range(time_steps):
        # Update the positions of the humans and zombies
        for human in humans:
            human.update_position()
        for zombie in zombies:
            zombie.update_position()

        # find the susceptible humans based on position and catch probability
        susceptible_humans = 0
        for human in humans:
            for zombie in zombies:
                if abs(human.position - zombie.position) < 10 and np.random.binomial(1, catch_probability(zombie, human)):
                    susceptible_humans += 1
                    continue
        
        # Calculate the number of infections
        new_infections = np.random.binomial(susceptible_humans, infection_rate * infected_population[-1] / total_population)

        # Calculate the number of killed zombies
        killed_zombies = np.random.binomial(len(zombies), killing_rate * healthy_population[-1] / total_population)

        # Update the healthy and infected populations
        healthy_population.append(healthy_population[-1] - new_infections)
        infected_population.append(infected_population[-1] + new_infections - killed_zombies)
        
        # Create new zombies
        zombies.extend([human for human in humans[:new_infections] if setattr(human, 'infected', True)])
        
        # Turn infected humans into zombies
        humans = humans[new_infections:]
        
        # Remove killed zombies
        zombies = zombies[killed_zombies:]

    return healthy_population, infected_population, humans, zombies

# Run simulations
healthy_results = []
infected_results = []
humans = []
zombies = []

for _ in range(num_simulations):
    healthy_population, infected_population, humans, zombies = run_simulation()
    healthy_results.append(healthy_population)
    infected_results.append(infected_population)
    humans.extend(humans)
    zombies.extend(zombies)

# Calculate averages
average_healthy = np.mean(healthy_results, axis=0)
average_infected = np.mean(infected_results, axis=0)

healthy_results = np.array(healthy_results).T
infected_results = np.array(infected_results).T

# Plot the results
plt.plot(healthy_results, color="blue", alpha=0.1)
plt.plot(average_healthy, label="Healthy Population")
plt.plot(infected_results, color="red", alpha=0.1)
plt.plot(average_infected, label="Infected Population")
plt.xlabel("Time steps")
plt.ylabel("Number of Individuals")
plt.title("Monte Carlo Simulation of a Zombie Apocalypse")
plt.legend()
plt.show()

# Plot the speed distributions of zombies and humans
zombie_speeds = [z.speed for z in zombies]
human_speeds = [h.speed for h in humans]
sns.kdeplot(zombie_speeds, label='Zombies', fill=True, color='red')
sns.kdeplot(human_speeds, label='Humans', fill=True, color='blue')
plt.xlabel('Speed')
plt.ylabel('Density')
plt.title('Speed Distributions')
plt.legend()
plt.show()

# Calculate the number of infected humans at the end of each simulation
num_infected = infected_results[-1]

# Plot the histogram of the number of infected humans at the end of the outbreak across trials
plt.hist(num_infected, bins=20, density=True)
plt.xlabel('Number of infected humans')
plt.ylabel('Probability')
plt.title('Number of Infected Humans at the End of Outbreak')
plt.show()

# Plot the number of infected humans at each step across trials
for i in range(num_simulations):
    plt.plot(infected_results[:, i], color='grey', alpha=0.1)
plt.plot(average_infected, color='black', linewidth=3)
plt.xlabel('Time steps')
plt.ylabel('Number of infected humans')
plt.title('Number of Infected Humans at Each Time Step Across Trials')
plt.show()


"""
Certainly! To understand how Temporal Difference (TD) learning can be viewed as a combination of Monte Carlo (MC) methods and Dynamic Programming (DP), we first need to understand the essence of each approach:

1. *Monte Carlo (MC) Methods*: 
   - MC methods estimate the value of a state based on the return (i.e., the sum of rewards) obtained from full episodes. You don't need a model of the environment, which means it's model-free.
   - It only updates values after the episode has ended.
   - It uses the actual returns from episodes to make updates.
   
2. *Dynamic Programming (DP) Methods*:
   - DP methods rely on the Bellman equation and need a model of the environment. They use the current estimate to update the value of a state. That is, they bootstrap from other estimates.
   - They update values at each step based on estimated returns, not actual returns.

*Temporal Difference (TD) Learning*:
   - Like MC, TD is model-free; it doesn't require a complete model of the environment.
   - Like DP, TD uses bootstrapping; it updates estimates based on other current estimates.
   - Unlike MC, which waits until the end of an episode, TD updates the value estimate at every step.
   - The key idea is that it uses the difference (or "temporal difference") between estimates at two successive time steps to update values.

To put it succinctly:

*MC* is like saying: "I'll wait until the end of the episode, see what the actual outcome is, and then decide if my initial prediction was good or not."

*DP* is like saying: "I won't wait; I'll use my current estimates to update my predictions right now."

*TD* combines the strengths of both: "I won't wait until the end like MC, but I also won't rely on a full model like DP. I'll just use my current experience and current estimates to update my predictions now."

This hybrid nature of TD, using aspects of both MC and DP, makes it suitable for many reinforcement learning problems where a balance between bias (from bootstrapping) and variance (from waiting for actual returns) is needed.
"""

"""
Temporal Difference (TD) learning can be used for a zombie apocalypse simulation in several ways, depending on the objective of the simulation. Here's a general outline of how TD might be used in such a context:

1. *State Representation*: 
    - Each state in the simulation could represent a specific situation during the zombie apocalypse. This could be a combination of the number of zombies in proximity, available weapons, health status, number of fellow survivors, location (e.g., outside, in a building), and more.

2. *Actions*: 
    - Actions could be decisions like "search for weapons", "hide", "run", "fight", "barricade", "search for food", etc.

3. *Rewards*: 
    - Immediate rewards (or penalties) are given for each action. For instance, successfully hiding from zombies might result in a small positive reward, while getting injured might result in a penalty.

4. *Policy*:
    - The goal would be to learn a policy that maximizes the expected cumulative reward, which might represent the survival duration or overall well-being during the apocalypse.

5. *TD Learning*: 
    - As the simulation progresses, TD can be used to update the value of each state based on the rewards received and the difference between expected and actual outcomes. For instance, if a particular location (state) is deemed safe but turns out to have many zombies, the value of that state would be adjusted downwards.

6. *Exploration and Exploitation*:
    - The agent (or group of survivors) needs to balance between exploring new strategies (to find better means of survival) and exploiting known strategies (that have worked well so far). Techniques like ε-greedy can be used for this.

7. *Scalability with Function Approximation*:
    - If the state space is very large, you might not be able to use tabular TD. In this case, you can use function approximation methods, like neural networks, to generalize across states. This way, experiences in one state can inform values of other similar states.

8. *Extensions*:
    - You can make the simulation more complex and interesting by adding other factors. For instance:
      - Multi-agent learning where multiple survivors (agents) learn to cooperate (or compete).
      - More complex state dynamics like the spread rate of the zombie infection, changing weather conditions affecting visibility and mobility, or dwindling supplies.

By running this simulation with TD learning, you'd expect to see the agent(s) evolve increasingly effective strategies to survive in a zombie-infested world. It would be fascinating to observe the kind of strategies that emerge and compare them to popular strategies depicted in media and literature about zombie apocalypses.
"""

"""
Temporal Difference update at each step so more info
Monte Carlo has no bias of the middle steps

Temporal difference = monte carlo + dynamic programming
"""
