



from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

CITY_SIZE = 10000 # Size of the city (in meters)
NUM_ZOMBIES = 10 # Number of zombies in the city
NUM_HUMANS = 100 # Number of humans in the city

class Person:
    def __init__(self, speed_mean, speed_std):
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.speed = np.random.normal(self.speed_mean, self.speed_std)
        self.position = np.random.uniform(0, CITY_SIZE)

    def update_position(self):
        self.position += self.speed
        if self.position < 0:
            self.position = 0
        elif self.position > CITY_SIZE:
            self.position = CITY_SIZE

class Zombie(Person):
    def __init__(self, speed_mean, speed_std):
        super().__init__(speed_mean, speed_std)
        self.color = 'red'
        self.speed = np.random.normal(self.speed_mean, self.speed_std) * 0.8 # Zombies are slower than humans
        self.infected = True

class Human(Person):
    def __init__(self, speed_mean, speed_std):
        super().__init__(speed_mean, speed_std)
        self.color = 'blue'
        self.infected = False

def catch_probability(zombie, human):
    z = (human.speed_mean - zombie.speed_mean) / np.sqrt(human.speed_std**2 + zombie.speed_std**2)
    p = 0.5 * (1 + erf(z / np.sqrt(2)))
    return p

def simulate_outbreak(num_steps):
    # Create zombies and humans
    zombies = [Zombie(3.0, 1.0) for _ in range(NUM_ZOMBIES)]
    humans = [Human(5.0, 1.5) for _ in range(NUM_HUMANS)]
    
    # Store the speeds of zombies and humans at each step
    zombie_speeds = np.zeros((num_steps, NUM_ZOMBIES))
    human_speeds = np.zeros((num_steps, NUM_HUMANS))

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
                if abs(zombie.position - human.position) < 10:
                    human.infected = True
                        
    # Calculate the number of infected humans
    num_infected = 0
    for human in humans:
        if human.infected:
            num_infected += 1
            
    return num_infected, zombie_speeds, human_speeds

def main():
    # Simulate the outbreak 1000 times
    num_steps = 100
    num_trials = 100
    infected = []
    zombie_speeds_all = []
    human_speeds_all = []
    catch_probs = []
    for trial in range(num_trials):
        inf, zombie_speeds, human_speeds = simulate_outbreak(num_steps)
        infected.append(inf)
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

        
    # Plot the histogram of the number of infected humans
    plt.hist(infected, bins=range(0, NUM_HUMANS + 1), density=True)
    # Plot the theoretical probability of catching a human
    x = np.linspace(0, NUM_HUMANS, 1000)
    y = norm.cdf(x, loc=NUM_ZOMBIES, scale=sqrt(NUM_ZOMBIES * (1 - catch_probs[-1]) + NUM_HUMANS * catch_probs[-1]))
    plt.plot(x, y, color='black')
    plt.xlabel('Number of infected humans')
    plt.ylabel('Probability')
    plt.show()
    
    
if __name__ == '__main__':
    main()


