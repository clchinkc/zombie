
# Description: A predator-prey model of a zombie outbreak

import matplotlib.pyplot as plt

# Import libraries
import numpy as np
from scipy.integrate import odeint

# Susceptible-Infected-Recovered model

# Initial conditions
H0 = 100000  # Initial number of susceptible individuals
Z0 = 100    # Initial number of infected individuals
D0 = 0    # Initial number of recovered individuals
Total = H0 + Z0 + D0 # Total population

# Parameters
a = 0.0 # birth rate of humans
b = 0.0001 # death rate of humans due to zombies
c = 0.00001 # death rate of humans due to overpopulation
d = 0.99 # rate at which zombies are created by turning humans
# db = birth rate of zombies due to humans
e = 0.001 # death rate of zombies due to natural causes
f = 0.00001 # death rate of zombies due to overpopulation

# Time span
time = np.linspace(0, 20, 1000)

# Differential equations
def HZD_model(HZD, time, Total, a, b, c, d, e, f):
    H, Z, D = HZD
    dHdt = a*H - b*H*Z - c*H**2
    dZdt = d*b*H*Z - e*Z - f*Z**2
    dDdt = (1-d)*b*H*Z + c*H**2  + e*Z + f*Z**2
    assert abs(dHdt + dZdt + dDdt - a*H) < 1e-10
    return dHdt, dZdt, dDdt

# Solve the system
sol_HZD = odeint(HZD_model, [H0, Z0, D0], time, args=(Total, a, b, c, d, e, f))
H, Z, D = sol_HZD.T

# Plot the results
plt.plot(time, H, label='Humans')
plt.plot(time, Z, label='Zombies')
plt.plot(time, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('HZD model of a Zombie Outbreak')
plt.legend()
plt.show()

# The rate of vaccine is 5% of susceptible individuals and people stay vaccined for 3 days on average. Find the number of people with vaccine immunity as a function of time.
ha = 0.05 *b*H*Z
h = ha
for i in range(1, 3):
    h += np.insert(ha, 0, np.zeros(i))[:-i]
plt.plot(time, H)
plt.plot(time, Z)
plt.plot(time, D)
plt.plot(time, h)
plt.legend()
plt.show()