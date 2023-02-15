# Brownian motion

"""
The Langevin equation is a stochastic differential equation that describes the behavior of a particle undergoing Brownian motion. Brownian motion is a random, continuous movement of particles suspended in a fluid, caused by their collision with the rapidly moving molecules of the fluid. The Langevin equation describes the evolution of the position and velocity of a Brownian particle over time, taking into account the combined effects of thermal fluctuations and frictional forces.

The equation is named after the French physicist Paul Langevin, who developed it in the early 20th century. It has the form:

dx/dt = v
dv/dt = -γv + ξ(t)

where x is the position of the particle, v is its velocity, γ is a constant that represents the strength of the frictional force, and ξ(t) is a random term that models the thermal fluctuations. The thermal fluctuations are often modeled as white noise, which is a random process with zero mean and constant variance.

The solution of the Langevin equation gives the probability distribution of the position and velocity of the Brownian particle at any given time, which can be used to calculate various statistical properties of the motion, such as the mean displacement, mean square displacement, and autocorrelation function. These properties have been extensively studied and form the basis of much of our understanding of Brownian motion and its applications in fields such as physics, chemistry, biology, and finance.

"""

# with: differentiable move

# https://github.com/DelSquared/Brownian-Motion


"""
import matplotlib.pyplot as plt
import numpy as np

# Parameters
gamma = 0.1  # friction coefficient
dt = 0.01  # time step
T = 10  # total time
n_steps = int(T / dt)  # number of steps
x0 = 0  # initial position
v0 = 1  # initial velocity

# Initialize arrays to store position and velocity
x = np.zeros(n_steps)
v = np.zeros(n_steps)
x[0] = x0
v[0] = v0

# Simulate Brownian motion using the Langevin equation
for i in range(1, n_steps):
    v[i] = v[i-1] - gamma * v[i-1] * dt + np.sqrt(2 * gamma * dt) * np.random.normal()
    x[i] = x[i-1] + v[i] * dt

# Plot the results
plt.plot(x, label='position')
plt.plot(v, label='velocity')
plt.xlabel('Time step')
plt.legend()
plt.show()
"""


"""
import matplotlib.pyplot as plt
import numpy as np

k = 20 # spring constant
m = 1 # mass

def R(t):
    return np.random.normal(0, 1, size=(2,))

def Langevin(xv, t, N):
    x, v = xv[:2], xv[2:] # current position and velocity
    dx = v
    dv = -m * v + k * N
    return np.concatenate((dx, dv))

def integrator3(f, xo, t, N):
    x = np.zeros(shape=(xo.shape[0], t.shape[0]))
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0]-1):
        k1 = f(x[:, i], t[i], N[i])
        k2 = f(x[:, i]+k1*dt/2, t[i]+dt/2, N[i])
        x[:, i+1] = x[:, i]+f(x[:, i]+k2*dt/2, t[i]+dt/2, N[i])*dt
    return x

t = np.linspace(0, 100, 100000) # time
N = np.random.normal(0, 1, size=(2, t.shape[0])) # noise
XV = np.array([0, 0, 1, 1]) # initial position and velocity

xv = integrator3(Langevin, XV, t, N.T)

fig = plt.figure()
plt.plot(xv[0, :], xv[1, :], label='position', color='blue')
plt.plot(xv[2, :], xv[3, :], label='velocity', color='red')
plt.title("Brownian motion")
plt.show()
"""
"""
import matplotlib.pyplot as plt
import numpy as np

k = 20
m = 1
w = 2
g = 0.1


def Oscillator(z_real, z_imag, t):
    z = z_real + z_imag * 1j
    dz = 1j * (w + g * np.random.normal(0, 1)) * z
    return np.real(dz), np.imag(dz)

def Langevin(xvz, t, N):
    x, v, z = xvz[:2], xvz[2:4], xvz[4:6]
    dx = v
    dv = -m * v + k * np.random.normal(0, 1, size=(2,))
    dz_real, dz_imag = Oscillator(z[0], z[1], t)
    return np.concatenate((dx, dv, [dz_real, dz_imag]))

t = np.linspace(0, 100, 100000)
XVZ = np.array([0, 0, 1, 1, 1, 1])
N = np.random.normal(0, 1, size=(2, t.shape[0]))

def integrator3(f, xo, t, N):
    x = np.zeros(shape=(xo.shape[0], t.shape[0]))
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0]-1):
        k1 = f(x[:, i], t[i], N[i])
        k2 = f(x[:, i]+k1*dt/2, t[i]+dt/2, N[i])
        x[:, i+1] = x[:, i]+f(x[:, i]+k2*dt/2, t[i]+dt/2, N[i])*dt
    return x

xvz = integrator3(Langevin, XVZ, t, N.T)

fig = plt.figure()
plt.plot(xvz[0, :], xvz[1, :], label='position', color='blue')
plt.plot(xvz[2, :], xvz[3, :], label='velocity', color='red')
plt.plot(xvz[4, :], xvz[5, :], label='oscillator', color='green')
plt.title("Brownian motion")
plt.show()
"""

import numpy as np
from matplotlib import pyplot as plt

# The intention of this solver is to eliminate the use of Scipy's odeint from this project. While it is a phenomenal tool
# its design makes it less than ideal for dealing with noisy/stochastic ODEs.

k = 5  # Set to 0 to disable noise for example


def ODE(x, t, noisemap):  # example ODE
    N = noisemap
    return np.array([1j * (1 + k * N) * x])


def integrator1(f, xo, t, N):  # Order 3
    x = np.zeros(shape=(xo.shape[0], t.shape[0]), dtype=complex)
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        x[:, i + 1] = x[:, i] + f(x[:, i], t[i], N[i]) * dt
    return x


def integrator2(f, xo, t, N):  # Order 3
    x = np.zeros(shape=(xo.shape[0], t.shape[0]), dtype=complex)
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        k1 = f(x[:, i], t[i], N[i])
        x[:, i + 1] = x[:, i] + f(x[:, i] + k1 * dt / 2, t[i] + dt / 2, N[i]) * dt
    return x


def integrator3(f, xo, t, N):  # Order 3
    x = np.zeros(shape=(xo.shape[0], t.shape[0]), dtype=complex)
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        k1 = f(x[:, i], t[i], N[i])
        k2 = f(x[:, i] + k1 * dt / 2, t[i] + dt / 2, N[i])
        x[:, i + 1] = x[:, i] + f(x[:, i] + k2 * dt / 2, t[i] + dt / 2, N[i]) * dt
    return x


def integrator4(f, xo, t, N):  # Order 3
    x = np.zeros(shape=(xo.shape[0], t.shape[0]), dtype=complex)
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        k1 = f(x[:, i], t[i], N[i])
        k2 = f(x[:, i] + k1 * dt / 2, t[i] + dt / 2, N[i])
        k3 = f(x[:, i] + k2 * dt / 2, t[i] + dt / 2, N[i])
        x[:, i + 1] = x[:, i] + f(x[:, i] + k3 * dt / 2, t[i] + dt / 2, N[i]) * dt
    return x


def integrator5(f, xo, t, N):  # Order 3
    x = np.zeros(shape=(xo.shape[0], t.shape[0]), dtype=complex)
    x[:, 0] = xo
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        k1 = f(x[:, i], t[i], N[i])
        k2 = f(x[:, i] + k1 * dt / 2, t[i] + dt / 2, N[i])
        k3 = f(x[:, i] + k2 * dt / 2, t[i] + dt / 2, N[i])
        k4 = f(x[:, i] + k3 * dt / 2, t[i] + dt / 2, N[i])
        x[:, i + 1] = x[:, i] + f(x[:, i] + k4 * dt / 2, t[i] + dt / 2, N[i]) * dt
    return x

t = np.linspace(0, 50, 5000)
N = np.random.normal(0, 1, size=(t.shape[0]))

xo = np.array([5 + 1j])

x1 = integrator1(ODE, xo, t, N)  # Trying all solvers for comparison
x2 = integrator2(ODE, xo, t, N)
x3 = integrator3(ODE, xo, t, N)
x4 = integrator4(ODE, xo, t, N)
x5 = integrator5(ODE, xo, t, N)

fig = plt.figure(figsize=((2**13) / 100, (2**10) / 100))  # Plotting
plt.plot(t, np.squeeze(np.real(x1)), label="real amplitude(1)")
plt.plot(t, np.squeeze(np.real(x2)), label="real amplitude(2)")
plt.plot(t, np.squeeze(np.real(x3)), label="real amplitude(3)")
plt.plot(t, np.squeeze(np.real(x4)), label="real amplitude(4)")
plt.plot(t, np.squeeze(np.real(x5)), label="real amplitude(5)")
plt.ylim(-10, 10)
plt.legend()
plt.show()
