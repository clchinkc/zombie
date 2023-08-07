
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# Parameters
zombie_growth_rate = 0.01
human_growth_rate = 0.005
interaction_rate = 0.02

# System Dynamics (linearized)
A = np.array([[zombie_growth_rate, interaction_rate],
              [-interaction_rate, human_growth_rate]])

B = np.array([[0.0],
              [1.0]])

# Cost Matrices
Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])

R = np.array([[10.0]])

# Solving the Algebraic Riccati Equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Simulation Parameters
dt = 0.01
time = np.arange(0, 10, dt)
zombies = np.zeros_like(time)
humans = np.zeros_like(time)
zombies[0] = 100
humans[0] = 200

# Simulating the Zombie Apocalypse
for t in range(1, len(time)):
    state = np.array([[zombies[t-1]], [humans[t-1]]])
    control = -K @ state
    dstate = A @ state + B @ control
    zombies[t] = zombies[t-1] + dstate[0, 0] * dt
    humans[t] = humans[t-1] + dstate[1, 0] * dt

# Plotting the Results
plt.plot(time, zombies, label='Zombies')
plt.plot(time, humans, label='Humans')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
