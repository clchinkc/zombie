
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

