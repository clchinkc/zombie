
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

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
        0, growth_rate, carrying_capacity, death_rate, emigration_rate)) # adding time t=0
    equilibrium_points = [point for point in equilibrium_points if np.isclose(
        dP_dt(point, 0, growth_rate, carrying_capacity, death_rate, emigration_rate), 0)]
    return equilibrium_points



def analyze_stability(P, growth_rate, carrying_capacity, death_rate, emigration_rate):
    """
    Analyze the stability and convergence by representing the differential equation as a matrix equation and use matrix algebra to find eigenvalues and eigenvectors and analyze the behavior of the system over time.
    """
    # Define the matrix
    A = np.array([[-growth_rate + death_rate + emigration_rate, 0], [growth_rate, 0]])

    # Find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the eigenvector associated with the eigenvalue of 0
    eigenvector = eigenvectors[:, np.isclose(eigenvalues, 0)]

    # Find the equilibrium points
    equilibrium_points = find_equilibrium_points(
        growth_rate, carrying_capacity, death_rate, emigration_rate)

    return eigenvalues, eigenvectors, eigenvector, equilibrium_points


def perform_sensitivity_analysis(t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate, parameter_values):
    """
    Perform a sensitivity analysis of the differential equation to identify the sensitivity of the system to changes in the assumptions or parameters.
    """
    sensitivity_results = []
    for growth_rate_list in parameter_values:
        sensitivity_solution = solve_ivp(dP_dt, [t0, t_final], [P0], args=(
            growth_rate_list, carrying_capacity, death_rate, emigration_rate))
        sensitivity_results.append(sensitivity_solution.y[0])
    return sensitivity_solution.t, sensitivity_results


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


def plot_equilibrium_points(equilibrium_points):
    plt.scatter(equilibrium_points, np.zeros(len(equilibrium_points)))
    plt.xlabel("Population size")
    plt.show()

def plot_sensitivity_analysis_results(t, sensitivity_results, parameter_values):
    for i, sensitivity_result in enumerate(sensitivity_results):
        plt.plot(t, sensitivity_result, label=f"growth rate = {parameter_values[i]}")
    plt.xlabel("Time")
    plt.ylabel("Population size")
    plt.legend()
    plt.show()

def plot_numerical_solution(t, P):
    plt.plot(t, P, label="Population size")
    plt.xlabel("Time")
    plt.ylabel("Population size")
    plt.legend()
    plt.show()

def plot_monte_carlo_simulation(t_values, P_values, mean_P, std_P):
    for P_value in P_values:
        plt.plot(t_values[0], P_value, color="gray", alpha=0.1)
    plt.plot(t_values[0], mean_P, label="Mean population size")
    plt.fill_between(t_values[0], mean_P-std_P, mean_P+std_P, alpha=0.25, label="Standard deviation")
    plt.xlabel("Time")
    plt.ylabel("Population size")
    plt.legend()
    plt.show()



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


# Find equilibrium points and perform analyses
equilibrium_points = find_equilibrium_points(growth_rate, carrying_capacity, death_rate, emigration_rate)
print("Equilibrium points are the population sizes at which the population neither grows nor shrinks. For this system, the equilibrium points are:", equilibrium_points)

# Analyze the stability of the system
eigenvalues, eigenvectors, eigenvector, equilibrium_points = analyze_stability(
    P0, growth_rate, carrying_capacity, death_rate, emigration_rate)
print("\n\nStability Analysis:\nEigenvalues and eigenvectors provide insights into the behavior of the system. Here are the results:")

for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
    print(f"\nEigenvalue {i}: {eigenvalue}")
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

print("\nPlotting the equilibrium points:")
plot_equilibrium_points(equilibrium_points)

print("\nPerforming a sensitivity analysis to identify how sensitive the system is to changes in the growth rate:")
t, sensitivity_results = perform_sensitivity_analysis(
    t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate, parameter_values)
print("\nSensitivity results:", sensitivity_results)

print("\nPlotting the results of the sensitivity analysis:")
plot_sensitivity_analysis_results(t, sensitivity_results, parameter_values)

print("\nGenerating a numerical solution of the differential equation to simulate the population over time:")
t, P = generate_numerical_solution(
    t0, t_final, P0, growth_rate, carrying_capacity, death_rate, emigration_rate)
print("Time points:", t)
print("Population size:", P)

print("\nPlotting the numerical solution:")
plot_numerical_solution(t, P)

print("\nRunning a Monte Carlo simulation to observe the variability in the population size due to variability in the parameters and initial conditions:")
t_values, P_values, mean_P, std_P = run_monte_carlo_simulation(t0, t_final, growth_rate, carrying_capacity, death_rate, emigration_rate, num_simulations, growth_rate_bounds, carrying_capacity_bounds, death_rate_bounds, emigration_rate_bounds, P0_bounds)

print("\nPlotting the results of the Monte Carlo simulation:")
plot_monte_carlo_simulation(t_values, P_values, mean_P, std_P)


