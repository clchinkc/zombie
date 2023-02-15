
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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as smp

# Generate Random Variables According to a Distribution
x = np.linspace(0,3,100)
f = 2*np.exp(-2*x)
F = 1-np.exp(-2*x)

Us = np.random.rand(10000)
F_inv_Us = -np.log(1-Us)/2

plt.title('Monte Carlo Simulation of $F^{-1}(u)$')
plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(F_inv_Us, histtype='step', color='red', density='norm', bins=100, label='$F^{-1}(u)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.show()

# Using a Search Sort Algorithm
x, y, F1, F2, E1, E2 = smp.symbols('x y F_1 F_2 E_1 E_2', real=True, positive=True)
fs = F1*smp.exp(-smp.sqrt(x/E1)) + F2*smp.exp(-smp.sqrt(x/E2))

Fs = smp.integrate(fs, (x,0,y)).doit()

Fn = smp.lambdify((y, E1, E2, F1, F2), Fs)
fn = smp.lambdify((x, E1, E2, F1, F2), fs)

E1 = E2 = 0.2
F1 = 1.3
F2 = 1.4
x = np.linspace(0,5,1000)
f = fn(x, E1, E2, F1, F2)
F = Fn(x, E1, E2, F1, F2)

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.plot(x, F, label=r'$F(x)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.show()

F_inv_Us = x[np.searchsorted(F[:-1], Us)]

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(F_inv_Us, histtype='step', color='red', density='norm', bins=100, label='$F^{-1}(u)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.xlim(0,2)
plt.show()

# Built In Random Variables
r = np.random.rayleigh(size=1000)

plt.figure(figsize=(8,3))
plt.hist(r, bins=100)
plt.show()

# Use These Random Variables To Conduct an Experiment
N = 100000

# Part 1 
X = np.random.poisson(lam=4, size=N)

# Part 2
x = np.linspace(0,5,1000)
F = Fn(x, E1, E2, F1, F2)
Us = np.random.rand(X.sum())
E = x[np.searchsorted(F[:-1], Us)]

idx = np.insert(X.cumsum(), 0, 0)[:-1]
idx[0:10]

E[0:10]

E_10s = np.add.reduceat(E, idx)

plt.figure(figsize=(5,3))
plt.hist(E_10s, bins=100)
plt.xlabel('Energy [GeV]', fontsize=20)
plt.ylabel('# Occurences')
plt.show()

print(np.mean(E_10s))
print(np.sum(E_10s>7.5)/len(E_10s))


