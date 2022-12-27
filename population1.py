
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math
import random

"""
Differential equations: One way to model the evolution of a population is to use differential equations, which describe how the population changes over time in response to various factors such as birth rate, death rate, and immigration. For example, the logistic equation is a commonly used differential equation model for population growth. To implement a differential equation model in Python, you could use a library like scipy to solve the equation numerically. Here's an example of how you could implement the logistic equation in Python:
"""


def logistic(y, t, r, K):
    # The logistic equation
    dy = r * y * (1 - y / K)
    return dy


# Initial population size
y0 = 10
# Time points
t = np.linspace(0, 5, 50)
# Growth rate and carrying capacity
r = 0.5
K = 50

# Solve the differential equation
solution = odeint(logistic, y0, t, args=(r, K))

# Plot the solution
plt.plot(t, solution)
plt.xlabel('Time')
plt.ylabel('Population size')
plt.show()


"""
Matrix algebra: Another way to model population dynamics is to use matrix algebra, which allows you to represent the transitions between different states of the population (e.g. births, deaths, immigration) as matrix operations. For example, the Leslie matrix is a commonly used matrix model for population growth. To implement a matrix model in Python, you could use a library like numpy to perform matrix operations. Here's an example of how you could implement the Leslie matrix in Python:
"""

# Initial population size
y0 = np.array([10, 20, 30])
# Transition matrix
L = np.array([[0, 0.5, 0.2], [0.3, 0, 0.1], [0.1, 0.2, 0]])

# Compute population size at each time point
for t in range(5):
    y0 = L @ y0

print(y0)

"""
Computer simulation: Another way to model population dynamics is to use computer simulation, which allows you to simulate the evolution of the population over time by iterating through a series of steps. For example, you could use a Monte Carlo simulation to model the spread of a disease through a population. To implement a simulation in Python, you could use a library like numpy to generate random numbers and a loop to iterate through the simulation steps. Here's an example of how you could implement a simple Monte Carlo simulation in Python:
"""

# Initial population size and number of time steps
N = 1000
T = 100

# Probability of infection and recovery
p_infect = 0.01
p_recover = 0.05

# Initialize the population
pop = np.zeros(N, dtype=int)

# Run the simulation
for t in range(T):
    # Infect a random subset of the population
    infected = np.random.rand(N)


"""
Logistic growth can be used to model the zombie number growth. It grows exponentially at the start due to the lack of defence of the survivors and slows down when the number of survivors decreases and they get protection.
"""
