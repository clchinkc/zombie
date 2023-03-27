# monte carlo

"""
Monte Carlo simulation can be used in a simulation of a zombie apocalypse in a number of ways. For example:
Modeling the spread of the zombie virus: Monte Carlo simulation can be used to model the random movement and interactions of people in a population, including the spread of the zombie virus from infected individuals to healthy ones.
Modeling the decision-making of survivors: Monte Carlo simulation can also be used to model the decision-making of survivors as they try to evade zombies and find safety. For example, it can simulate the probability of a survivor successfully reaching a safe haven given various factors such as their starting location, the number of zombies in the area, and their supplies.
Modeling the impact of interventions: Monte Carlo simulation can be used to model the impact of various interventions on the spread of the zombie virus and the survival of the human population. For example, it can simulate the effect of a quarantine on the number of new zombie infections or the impact of a vaccine on the number of survivors.
Generating scenarios: Monte Carlo simulation can be used to generate a range of possible scenarios for a zombie apocalypse, taking into account the uncertainty and randomness inherent in such a situation. This can provide valuable insights into the range of possible outcomes and help decision-makers prepare for a wide range of contingencies.
However, it's important to note that the results of a Monte Carlo simulation are only as good as the model used, and it's crucial to carefully validate the assumptions and inputs used in the simulation.
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
"""
The Monte Carlo method is a model-free reinforcement learning algorithm that estimates the value of a state by averaging the returns observed after visiting the state. It learns by simulating many episodes and updating the value of states based on the observed rewards. In this specific code, the Monte Carlo e-soft method can be used to find an optimal policy for the specific environment. The algorithm first creates a random policy and a state-action dictionary. Then, for a given number of episodes, it runs a game using the current policy and records the states, actions, and rewards obtained. It then updates the state-action values using the observed returns and uses epsilon-greedy exploration to update the policy. The test_policy function is used to evaluate the performance of the learned policy by running a given number of games and returning the fraction of wins.
"""

# without: Markov chain Monte Carlo, Markov Decision Processes, https://pojenlai.wordpress.com/2018/11/14/paper-reading-the-naive-utility-calculus-joint-inferences-about-the-costs-and-rewards-of-actions/

# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/montecarlo1.ipynb



import matplotlib.pyplot as plt
import numpy as np

# Parameters
num_simulations = 1000
time_steps = 500
initial_healthy = 1000
initial_infected = 10
infection_rate = 0.1
killing_rate = 0.05

# Function to run one simulation
def run_simulation():
    healthy_population = [initial_healthy]
    infected_population = [initial_infected]
    total_population = initial_healthy + initial_infected

    for _ in range(time_steps):
        # Calculate the number of infections
        new_infections = np.random.binomial(healthy_population[-1], infection_rate * infected_population[-1] / total_population)

        # Calculate the number of killed zombies
        killed_zombies = np.random.binomial(infected_population[-1], killing_rate)

        # Update the healthy and infected populations
        healthy_population.append(healthy_population[-1] - new_infections)
        infected_population.append(infected_population[-1] + new_infections - killed_zombies)

    return healthy_population, infected_population

# Run simulations
healthy_results = []
infected_results = []

for _ in range(num_simulations):
    healthy_population, infected_population = run_simulation()
    healthy_results.append(healthy_population)
    infected_results.append(infected_population)

# Calculate averages
average_healthy = np.mean(healthy_results, axis=0)
average_infected = np.mean(infected_results, axis=0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(average_healthy, label="Healthy Population")
plt.plot(average_infected, label="Infected Population")
plt.xlabel("Time steps")
plt.ylabel("Number of Individuals")
plt.legend()
plt.title("Monte Carlo Simulation of a Zombie Apocalypse")
plt.show()


