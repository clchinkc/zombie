
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import math
import random


"""
Matrix algebra: Another way to model population dynamics is to use matrix algebra, which allows you to represent the transitions between different states of the population (e.g. births, deaths, immigration) as matrix operations. For example, the Leslie matrix is a commonly used matrix model for population growth. Here is an example of how a Leslie matrix could be used to represent the change of population size during a zombie apocalypse:
"""

def model_population_size(P0, L, t0, t_final):
    """
    Model the population size during a zombie apocalypse using a matrix algebra model and a differential equation.
    """
    # Define the differential equation
    def dP_dt(t, P, L):
        return L @ P

    # Generate a numerical solution of the differential equation
    solution = solve_ivp(dP_dt, [t0, t_final], P0, args=(L,))

    # Extract the population size at each time point
    t = solution.t
    P = solution.y

    return t, P

# Initial population size
P0 = np.array([10000, 1000, 1000])

# Transition matrix
L = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

# Initial time
t0 = 0

# Set the time period for the simulation (in days)
t_final = 365

# Model the population size
t, P = model_population_size(P0, L, t0, t_final)

print(P)


"""
This code defines an initial population size as a NumPy array of three values [10000, 1000, 1000], which represents the number of humans, zombies, and dead bodies in the population. It also defines a transition matrix as a 3x3 NumPy array [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]], which represents the probability of transitions between the different states (human, zombie, or dead).
The code then uses a loop to compute the population size at each time point by iteratively multiplying the initial population size by the transition matrix. At the end of the simulation, the final population size will be printed to the console.
This matrix algebra model is a simplified representation of the population size during a zombie apocalypse, but it can still provide insight into the behavior of the population over time. For example, you could use this model to understand how the number of humans, zombies, and dead bodies in the population changes over time, and to identify any trends or patterns in the data.
"""

"""
Logistic growth can be used to model the zombie number growth. It grows exponentially at the start due to the lack of defence of the survivors and slows down when the number of survivors decreases and they get protection.
"""

"""
One potential advantage of using differential equations to model a zombie apocalypse is that they can provide a detailed, continuous description of the evolution of the system over time. This can allow for a more accurate prediction of the behavior of the population, as well as the identification of any critical thresholds or tipping points that might occur. However, solving differential equations can be mathematically challenging, and the model may be sensitive to changes in the assumptions or parameters that are used.
Matrix algebra can also be used to model a zombie apocalypse, and has the advantage of being able to represent the system in a compact and easily manipulable form. Matrix algebra can also be used to analyze the stability and convergence of the system, which can be useful for predicting the long-term behavior of the population. However, matrix algebra is generally less detailed than differential equations, and may not be able to capture all of the nuances of the system.
Computer simulations can be a powerful tool for modeling a zombie apocalypse, as they can allow for the incorporation of a wide range of factors and variables into the model. Computer simulations can also be run quickly and easily, allowing for the exploration of different scenarios and sensitivity analyses. However, computer simulations rely on the accuracy and validity of the underlying mathematical models, and may not always be able to capture all of the complexity of the system.

The first approach for combining differential equations, matrix algebra, and computer simulations to model a zombie apocalypse would involve the following steps:
Use differential equations to describe the detailed, continuous evolution of the system over time. This might involve defining a set of differential equations that represent the various factors that can affect the population, such as birth rates, death rates, immigration, and emigration.
Use matrix algebra to analyze the stability and convergence of the system. This might involve representing the differential equations as a matrix equation, and using matrix operations to analyze the behavior of the system over time. This can help to identify any critical thresholds or tipping points that might occur, as well as the long-term behavior of the population.
Incorporate the results of the matrix algebra analysis into a computer simulation. This might involve using the results of the matrix algebra analysis to specify the initial conditions and parameters of the simulation, as well as the rules for how the system evolves over time.
Use the computer simulation to explore different scenarios and sensitivity analyses. This might involve running the simulation under different sets of assumptions and parameters, and analyzing the results to see how they compare to the predictions of the differential equations and matrix algebra.
Overall, this approach combines the strengths of differential equations, matrix algebra, and computer simulations in order to provide a detailed, continuous description of the evolution of the system over time, as well as the ability to explore different scenarios and sensitivity analyses.
"""

"""
Here is an example of how the first approach for combining differential equations, matrix algebra, and computer simulations could be implemented in Python:
"""


def dP_dt(P, t, growth_rate, carrying_capacity, death_rate, emigration_rate):
    """
    A differential equation called logistic growth that describes how the population size of a species can change over time.
    """
    real_growth_rate = growth_rate * (1 - P/carrying_capacity)
    real_death_rate = death_rate * P
    real_emigration_rate = emigration_rate * P
    return real_growth_rate - real_death_rate - real_emigration_rate


def find_equilibrium_points(growth_rate, carrying_capacity, death_rate, emigration_rate):
    """
    Find the equilibrium points of the differential equation, and identify stable and unstable equilibrium points and oscillations.
    """
    equilibrium_points = fsolve(dP_dt, [-1000, -100, -10, -1, 0, 1, 10, 100, 1000], args=(
        growth_rate, carrying_capacity, death_rate, emigration_rate))
    equilibrium_points = [point for point in equilibrium_points if np.isclose(
        dP_dt(point, 0, growth_rate, carrying_capacity, death_rate, emigration_rate), 0)]
    return equilibrium_points


def analyze_stability(P, growth_rate, carrying_capacity, death_rate, emigration_rate):
    """
    Analyze the stability and convergence by representing the differential equation as a matrix equation and use matrix algebra to find eigenvalues and eigenvectors and analyze the behavior of the system over time.
    """
    A = np.array(
        [[-death_rate, -emigration_rate, growth_rate * (1 - P/carrying_capacity)]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def perform_sensitivity_analysis(t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate, parameter_values):
    """
    Perform a sensitivity analysis of the differential equation to identify the sensitivity of the system to changes in the assumptions or parameters.
    """
    sensitivity_results = []
    for growth_rate_list in parameter_values:
        sensitivity_solution = solve_ivp(dP_dt, [t0, t_final], [P0], args=(
            growth_rate_list, carrying_capacity, death_rate, emigration_rate))
        sensitivity_results.append(sensitivity_solution.y[0])
    return sensitivity_results


def generate_numerical_solution(t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate):
    """
    Generate a numerical solution of the differential equation.
    """
    solution = solve_ivp(dP_dt, [t0, t_final], [P0], args=(
        growth_rate, carrying_capacity, death_rate, emigration_rate))
    return solution.t, solution.y[0]

# event in solve_ivp happen when number of either is 0
# a = solve_ivp(dP_dt, [t0, t_final], [P0], args=(growth_rate, carrying_capacity, death_rate, emigration_rate), events=(lambda t, y: y[0] - 0, lambda t, y: y[1] - 0))



def run_monte_carlo_simulation(t0, t_final, growth_rate, carrying_capacity, death_rate, emigration_rate, num_simulations, growth_rate_bounds, carrying_capacity_bounds, death_rate_bounds, emigration_rate_bounds, P0_bounds):
  """
  Run a Monte Carlo simulation of the population size.
  """
  # Initialize lists to store the results of the simulations
  t_values = []
  P_values = []

  # Run the Monte Carlo simulation
  for i in range(num_simulations):
    # Generate random samples of the parameters and initial conditions
    growth_rate_sample = random.uniform(*growth_rate_bounds)
    carrying_capacity_sample = random.uniform(*carrying_capacity_bounds)
    death_rate_sample = random.uniform(*death_rate_bounds)
    emigration_rate_sample = random.uniform(*emigration_rate_bounds)
    P0_sample = random.uniform(*P0_bounds)

    # Generate a numerical solution of the equation using the sampled parameters and initial conditions
    t, P = generate_numerical_solution(t0, t_final, P0_sample, growth_rate_sample, carrying_capacity_sample, death_rate_sample, emigration_rate_sample)

    # Store the results of the simulation
    t_values.append(t)
    P_values.append(P)

  # Analyze the results of the simulations
  mean_P = np.mean(P_values, axis=0)
  std_P = np.std(P_values, axis=0)
  return t_values, P_values, mean_P, std_P



# Define the initial conditions for the simulation
P0 = 100
t0 = 0
t_final = 100

# Define the parameters for the differential equation
growth_rate = 0.1
carrying_capacity = 1000
death_rate = 0.05
emigration_rate = 0.01

# Define the parameter values to use in the sensitivity analysis
parameter_values = [0.1, 0.5, 1, 2, 5]

# Set the number of simulations to run
num_simulations = 1000
# Set the bounds for the random samples
growth_rate_bounds = (0.01, 0.5)
carrying_capacity_bounds = (500, 1500)
death_rate_bounds = (0.01, 0.1)
emigration_rate_bounds = (0.01, 0.1)
P0_bounds = (50, 150)

# Find the equilibrium points
equilibrium_points = find_equilibrium_points(
    growth_rate, carrying_capacity, death_rate, emigration_rate)
print("Equilibrium points:", equilibrium_points)

# Analyze the stability of the system
eigenvalues, eigenvectors = analyze_stability(
    P0, growth_rate, carrying_capacity, death_rate, emigration_rate)
# Analyze the eigenvalues and eigenvectors to gain insights into the behavior of the system
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
    print(f"Eigenvalue {i}: {eigenvalue}")
    print(f"Eigenvector {i}: {eigenvector}")

    # Interpret the real part of the eigenvalue
    if eigenvalue.real > 0:
        print(f"Eigenvalue {i} has a positive real part, indicating that the system will tend to increase over time")
    elif eigenvalue.real < 0:
        print(f"Eigenvalue {i} has a negative real part, indicating that the system will tend to decrease over time")
    else:
        print(f"Eigenvalue {i} has a zero real part, indicating that the system will tend to remain constant over time")

    # Interpret the imaginary part of the eigenvalue
    if eigenvalue.imag != 0:
        print(f"Eigenvalue {i} has a non-zero imaginary part, indicating that the system will tend to oscillate or exhibit periodic behavior")

    # Interpret the eigenvector
    print(f"Eigenvector {i} points in the direction {eigenvector}, indicating that this is the direction in which the system is most likely to change")

"""
# Set the equilibrium initial population size
P_init = eigenvectors[:, np.argmax(eigenvalues)]
# Generate a numerical solution of the equation
t, P = generate_numerical_solution(t0, t_final, P_init, growth_rate, carrying_capacity, death_rate, emigration_rate)
"""

# Perform a sensitivity analysis
sensitivity_results = perform_sensitivity_analysis(
    t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate, parameter_values)
print("Sensitivity results:", sensitivity_results)

# Plot the sensitivity analysis results
for i, sensitivity_result in enumerate(sensitivity_results):
    plt.plot(t, sensitivity_result, label=f"growth rate = {parameter_values[i]}")
plt.xlabel("Time")
plt.ylabel("Population size")
plt.legend()
plt.show()

# Generate a numerical solution of the equation
t, P = generate_numerical_solution(
    t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate)
print("Time points:", t)
print("Population size:", P)

# Plot the numerical solution of the simulation
plt.plot(t, P, label="Population size")
plt.xlabel("Time")
plt.ylabel("Population size")
plt.legend()
plt.show()

# Run the Monte Carlo simulation
t_values, P_values, mean_P, std_P = run_monte_carlo_simulation(t0, t_final, growth_rate, carrying_capacity, death_rate, emigration_rate, num_simulations, growth_rate_bounds, carrying_capacity_bounds, death_rate_bounds, emigration_rate_bounds, P0_bounds)

# Plot the results of the monte carlo simulation
plt.plot(t_values, mean_P, label="Mean population size")
plt.plot(t_values, mean_P + std_P, label="Mean population size + standard deviation")
plt.plot(t_values, mean_P - std_P, label="Mean population size - standard deviation")
plt.xlabel("Time")
plt.ylabel("Population size")
plt.legend()
plt.show()


"""
The find_equilibrium_points function can help you identify the points at which the rate of change of the population size is zero, which can provide insight into the long-term behavior of the population.
The analyze_stability function can help you identify any stable or unstable equilibrium points, which can provide insight into the sensitivity of the system to changes in the assumptions or parameters.
The perform_sensitivity_analysis function can help you identify which factors have the greatest impact on the behavior of the population, and can provide insight into how the system might respond to different scenarios or interventions.
And the generate_numerical_solution function can help you generate a prediction of the population size over time, which can help to identify any trends or patterns in the data.
The run_monte_carlo_simulation function can be used to run a Monte Carlo simulation of the population size. This can help to incorporate uncertainty and randomness into the simulation, and can be especially useful for analyzing systems with complex behaviors or large numbers of interacting components.
"""
