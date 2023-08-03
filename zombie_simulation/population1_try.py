
import numpy as np
from scipy.integrate import solve_ivp

# Define the transition matrix for the stage-based model
# The order of the stages is: Healthy, Infected, Zombie, Dead
transition_matrix = np.array([[0.9, 0.1, 0.0, 0.0],  # From Healthy
                              [0.0, 0.6, 0.4, 0.0],  # From Infected
                              [0.0, 0.0, 0.5, 0.5],  # From Zombie
                              [0.0, 0.0, 0.0, 1.0]])  # From Dead

def model(t, y, alpha, beta, gamma, delta):
    H, I, Z, D = y

    # Now apply the differential equation model
    dHdt = -alpha * H * Z
    dIdt = alpha * H * Z - beta * I
    dZdt = beta * I - gamma * Z
    dDdt = gamma * Z

    rates = [dHdt, dIdt, dZdt, dDdt]

    # Apply the matrix-based model, using rates from the differential model
    next_stage = np.dot(transition_matrix, rates)

    return next_stage

# Initial conditions
initial_healthy = 500
initial_infected = 10
initial_zombie = 5
initial_dead = 0
y0 = [initial_healthy, initial_infected, initial_zombie, initial_dead]

# Time grid
t_span = (0, 100)

# Parameters for the model
alpha = 0.01
beta = 0.01
gamma = 0.02
delta = 0.03

# Solve the differential equations using solve_ivp
solution = solve_ivp(model, t_span, y0, args=(alpha, beta, gamma, delta), t_eval=np.linspace(t_span[0], t_span[1], 100))

# Extract the results
t = solution.t
y = solution.y

# y now contains the solution. Each column corresponds to a different stage
# and each row corresponds to a different time point.

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.plot(t, y[0], label='Healthy (H)')
plt.plot(t, y[1], label='Infected (I)')
plt.plot(t, y[2], label='Zombie (Z)')
plt.plot(t, y[3], label='Dead (D)')
plt.xlabel('Days since outbreak')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()





