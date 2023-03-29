



from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

CITY_SIZE = 10000 # Size of the city (in meters)
NUM_ZOMBIES = 10 # Number of zombies in the city
NUM_HUMANS = 100 # Number of humans in the city

class Person:
    def __init__(self, speed_mean, speed_std, position=None):
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.speed = np.random.normal(self.speed_mean, self.speed_std)
        self.position = position if position is not None else np.random.uniform(0, CITY_SIZE)

    def update_position(self):
        self.position += self.speed
        self.position = np.clip(self.position, 0, CITY_SIZE)

class Zombie(Person):
    def __init__(self, speed_mean, speed_std, position=None):
        super().__init__(speed_mean, speed_std, position)
        self.color = 'red'
        self.speed = np.random.normal(self.speed_mean, self.speed_std) * 0.8 # Zombies are slower than humans
        self.infected = True

class Human(Person):
    def __init__(self, speed_mean, speed_std, position=None):
        super().__init__(speed_mean, speed_std, position)
        self.color = 'green'
        self.infected = False

def catch_probability(zombie, human):
    z = (human.speed_mean - zombie.speed_mean) / np.sqrt(human.speed_std**2 + zombie.speed_std**2)
    p = 0.5 * (1 + erf(z / np.sqrt(2)))
    return p

def simulate_outbreak(num_steps):
    # Create zombies and humans
    zombies = [Zombie(1.0, 1.0) for _ in range(NUM_ZOMBIES)]
    humans = [Human(5.0, 1.5) for _ in range(NUM_HUMANS)]
    
    # Store the speeds of zombies and humans at each step
    zombie_speeds = np.zeros((num_steps, NUM_ZOMBIES))
    human_speeds = np.zeros((num_steps, NUM_HUMANS))
    
    # Store infected humans at each step
    infected = np.zeros((num_steps, NUM_HUMANS))

    # Simulate the positions of zombies and humans in the city through time
    for step in range(num_steps):
        # Update the positions of the zombies and humans
        for zombie in zombies:
            zombie.update_position()
            zombie_speeds[step, zombies.index(zombie)] = zombie.speed
        for human in humans:
            human.update_position()
            human_speeds[step, humans.index(human)] = human.speed

        # Check if any zombies have caught any humans
        for zombie in zombies:
            for human in humans:
                if abs(zombie.position - human.position) < 10 and np.random.binomial(1, catch_probability(zombie, human)):
                    human.infected = True
        
        # store the infected status of humans at each step
        infected[step] = np.array([human.infected for human in humans])
            
    return infected, zombie_speeds, human_speeds

def main():
    # Simulate the outbreak 1000 times
    num_steps = 100
    num_trials = 100
    infected = np.zeros((num_trials, num_steps, NUM_HUMANS))
    zombie_speeds_all = []
    human_speeds_all = []
    catch_probs = []
    for trial in range(num_trials):
        inf, zombie_speeds, human_speeds = simulate_outbreak(num_steps)
        infected[trial] = inf
        zombie_speeds_all.append(zombie_speeds)
        human_speeds_all.append(human_speeds)
        catch_probs.append(catch_probs[-1] if trial > 0 else 0)  # append last catch probability or 0 if first trial
        
    # Plot the speed distributions of zombies and humans across trials
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    zombie_speeds = np.concatenate(zombie_speeds_all)
    human_speeds = np.concatenate(human_speeds_all)
    zombie_avg_speed = np.mean(zombie_speeds, axis=0)
    human_avg_speed = np.mean(human_speeds, axis=0)
    sns.kdeplot(zombie_avg_speed, label='Zombies', fill=True)
    sns.kdeplot(human_avg_speed, label='Humans', fill=True)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distributions')
    plt.legend()
    plt.show()

    # Plot the histogram of the number of infected humans at the end of the outbreak across trials
    # infected.shape = (num_trials, num_steps, NUM_HUMANS)
    num_infected = np.sum(infected[:, -1, :], axis=1)
    plt.hist(num_infected, bins=range(0, NUM_HUMANS + 1), density=True)
    # Plot the theoretical probability of catching a human
    x = np.linspace(0, NUM_HUMANS, 1000)
    y = norm.cdf(x, loc=NUM_ZOMBIES, scale=sqrt(NUM_ZOMBIES * (1 - catch_probs[-1]) + NUM_HUMANS * catch_probs[-1]))
    plt.plot(x, y, color='black')
    plt.xlabel('Number of infected humans')
    plt.ylabel('Probability')
    plt.show()
    
    # Plot the number of infected humans at each step across trials
    infected = np.sum(infected, axis=2)
    plt.plot(infected.T, color='grey', alpha=0.1)
    plt.plot(np.mean(infected, axis=0), color='black', linewidth=3)
    plt.xlabel('Step')
    plt.ylabel('Number of infected humans')
    plt.show()
    
if __name__ == '__main__':
    main()


