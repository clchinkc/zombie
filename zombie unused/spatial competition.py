
# Natural Selection

"""
In a simulation of a zombie apocalypse, natural selection can be applied to the population of both humans and zombies, providing a basis for understanding how populations of zombies and humans might evolve over time in response to changing environmental conditions and selective pressures. For example:
The virus that causes zombies could evolve over time to become more or less virulent, or to affect different hosts in different ways. This evolution would be shaped by natural selection, as the most successful strains of the virus would be those that are able to infect and reproduce effectively.
The competition for resources between humans and zombies could lead to the evolution of adaptations that allow one species to outcompete the other. For example, humans might evolve to be faster or more efficient at finding food, or zombies might evolve to be better at detecting and attacking their prey.
Natural selection could also play a role in the development of resistance to the virus among survivors. Those who are immune to the virus would be more likely to survive and pass on their immunity to their offspring, while those who are not immune would be more likely to succumb to the infection. Over time, the proportion of the human population that is immune to the virus would increase.
By considering these factors, a simulation of a zombie apocalypse could provide a more nuanced and scientifically informed understanding of the dynamics of the outbreak and the likelihood of different outcomes.
"""
"""
Designing a natural selection simulation of a zombie apocalypse would require making a number of decisions about the specifics of the scenario, such as the initial conditions and the rules governing how humans and zombies interact. However, a possible approach could be as follows:
Start with a simple mathematical model for the populations of humans and zombies. This model could be a set of equations that describe how the populations change over time, taking into account factors such as births, deaths, and conversions from humans to zombies.
Add in parameters that describe the strengths of the various selective pressures acting on the populations. For example, the rate of zombie-human encounters, the probability of a human being infected and converted to a zombie, the mortality rate of zombies, and the mortality rate of humans.
Run the simulation over a period of time, starting with the initial conditions and updating the populations at regular intervals based on the mathematical model and the parameters.
Analyze the results of the simulation to see how the populations of humans and zombies evolve over time. This could include examining how the numbers of humans and zombies change, as well as how the parameters evolve over time.
Repeat the simulation with different parameters and initial conditions to explore the range of possible outcomes and to test the robustness of the results.
Refine the simulation as needed based on the results of the initial runs. This could include adjusting the parameters, adding or modifying equations, or incorporating additional factors that were not considered in the initial version of the simulation.
"""

# with: competition of resources between human and zombies
# without: predator and prey, unclear relationship between variables

# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid13.ipynb
# https://github.com/OwenDavisBower/Natural-Selection-Simulator-Python-/blob/master/natural_selection.py
# https://github.com/topics/natural-selection
# https://thekidshouldseethis.com/post/simulating-natural-selection
# https://pycon.org.il/2016/static/sessions/yoav-ram.pdf

"""
The spatiotemporal change for a species over time can be written as
∂ρ/∂t = D∇^2ρ + α(r)ρ - βρ^2
---
where
ρ(r,t) is the density (in numbers) of a species at position r and time t
ρ(r,0), the distribution of species at the initial time
---
D∇^2ρ = diffusion of a species (i.e. a group of animals spreading out over some biome)
D = diffusion rate
d1 and d2 is the diffusion rate
∇^2ρ(r) > 0 means average density around r is more dense
∇^2ρ(r) < 0 means average density around r is less dense
---
α(r)ρ = growth of the species with α(r) as the abundance of resources at point r
α = growth rate
---
βρ^2 = death of the species, through natural causes and competition
β = tolerance constant
---
when multiplying by β, one obtains this equation (all parameters follow from above):
∂u/∂t = D∇^2u + α(r)u - u^2
and in one dimension:
∂u/∂t = d(∂^2u/∂x^2) + u(r(x) - u) letting r(x) be the resources given in time x
---
Now suppose there are two species with normalized densities u and v. The equation above can be adjusted by adding two new values a and b: the relative competing strengths of each species. The equations are given by
∂u/∂t = d1(∂^2u/∂x^2) + u(r(x) - u - av)
∂v/∂t = d2(∂^2v/∂x^2) + v(r(x) - bu - v)
with initial conditions u(x,0) = f(x) and v(x,0) = g(x)
a and b is the species competitiveness
a < b means species 1 is better at outcompeting species 2
a > b means species 1 is worse at outcompeting species 2
---
The Question:
What makes one species better at outcompeting another species given a closed boundary with a zero flux condition? (i.e. finite area and animals cannot leave the area). Things to keep in mind
a and b
d1 and d2
f(x) and g(x)
---
The Finite Difference method
Instead of using dx and dt, we make a finite approximation with Δx and Δt, leading to
u(i,m+1) = u(i,m) + (d1Δt)/(Δx^2) (u(i,m+1) + u(i,m-1) - 2u(i,m)) + u(i,m)(r(i) - u(i,m) - av(i,m))Δt
v(i,m+1) = v(i,m) + (d2Δt)/(Δx^2) (v(i,m+1) + v(i,m-1) - 2v(i,m)) + v(i,m)(r(i) - bu(i,m) - v(i,m))Δt

i = position index
m = time index
"""

import matplotlib.pyplot as plt
import numba

# import libraries
import numpy as np
from matplotlib import animation, cm
from matplotlib.animation import PillowWriter

# set up initial conditions

times = np.linspace(0, 100, 100000)
x = np.linspace(-2, 2, 60)
p1 = np.zeros([len(times), len(x)])
p2 = np.zeros([len(times), len(x)])
r = np.exp(-x**2/0.1)
p1[0] = np.exp(-(x-1.2)**2/0.02)
p2[0] = np.exp(-(x+1.2)**2/0.02)

d1 = 0.01 # higher means diffuse faster
d2 = 0.1 # lower means diffuse slower
# if same competitiveness, lower diffusion rate will seem to lose but always catch from behind
a = 1 # higher means better competitiveness
b = 0.5 # lower means worse competitiveness
# high competitiveness always get more resources and grow faster
# equilibrium when same diffusion rate same competitiveness

dt = np.diff(times)[0]
dx = np.diff(x)[0]

plt.figure(figsize=(10,6))
plt.title("initial position", loc='center')
plt.plot(x, r, color='green', label='Resources')
plt.plot(x, p1[0], label='Species 1')
plt.plot(x, p2[0], label='Species 2')
plt.legend(loc='upper right')
plt.show()

print("dt/dx**2 =", dt/dx**2) # dt/dx**2 means how many times the species will diffuse in one time step

"""
This code appears to be a simulation of two species competing for a limited resource in a one-dimensional environment. The code sets up arrays to store the population densities of two species (p1 and p2) over time (times) and across space (x). The initial conditions of the resource (r) and the two species (p1[0] and p2[0]) are defined using Gaussian functions.
Next, the code sets some parameters that govern the simulation:
d1 and d2: The diffusion rates of the two species. A higher diffusion rate means that the species will diffuse (spread out) faster, and a lower diffusion rate means that the species will diffuse slower.
a and b: The competitiveness of the two species. A higher competitiveness means that the species will be better able to compete for resources, and a lower competitiveness means that the species will be worse at competing.
The simulation is run for a specified time interval times using the finite difference method to update the populations of the two species and the resource over time based on the differential equations that describe the evolution of the system.
Finally, the code plots the initial conditions of the resource and the two species using the matplotlib library.
It's important to note that this code only sets up the simulation and defines the initial conditions, and the actual simulation would need to be run using additional code to see how the populations evolve over time.
"""

# solve the equation and get the solution

@numba.jit("UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8[:])", nopython=True, nogil=True)
def solve_pop(p1, p2, r):
    for t in range(0,len(times)-1):
        for i in range(0, len(p1[0])):
            # Insulated Boundary Conditions
            if i==0:
                deriv2_1 = 2*p1[t][i+1] - 2*p1[t][i]
                deriv2_2 = 2*p2[t][i+1] - 2*p2[t][i]
            elif i==len(p1[0])-1:
                deriv2_1 =  2*p1[t][i-1] - 2*p1[t][i]
                deriv2_2 =  2*p2[t][i-1] - 2*p2[t][i]
            else:
                deriv2_1 = p1[t][i+1] + p1[t][i-1] - 2*p1[t][i]
                deriv2_2 = p2[t][i+1] + p2[t][i-1] - 2*p2[t][i]
                
            p1[t+1][i] = p1[t][i] + d1*dt/dx**2 * deriv2_1 + \
                         p1[t][i] * dt * (r[i] - p1[t][i] - a*p2[t][i])
            p2[t+1][i] = p2[t][i] + d2*dt/dx**2 * deriv2_2 + \
                         p2[t][i] * dt * (r[i] - p2[t][i] - b*p1[t][i])
            
    return p1, p2

p1, p2 = solve_pop(p1, p2, r)

"""
The code defines a function solve_pop that simulates the growth and competition of two populations, Species 1 and Species 2, for a given set of resources.
The function uses finite differences to model the diffusion and growth of the populations over time.
The populations compete for resources and their growth rates are determined by their competitiveness (parameters a and b) and their diffusion rates (parameters d1 and d2).
The boundaries of the simulation are insulated, meaning that the populations at the edge of the simulation space cannot diffuse out of the space.
The function returns the final populations after the simulation has run.
The simulation is accelerated using the numba library.
"""

# plot the solution

i= 10 # time step
plt.figure(figsize=(14,8))
plt.plot(x,p1[i], lw=3, label='Species 1, $d_1=${:.3f}'.format(d1))
plt.plot(x,p2[i], lw=3, label='Species 2, $d_2=${:.3f}'.format(d2))
plt.plot(x,r, ls='--', label='Resources $r(x)$')
plt.legend(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.grid()
plt.show()

# animate the solution

def animate(i):
    line1.set_data(x,p1[500*i])
    line2.set_data(x,p2[500*i])
    line3.set_data(x,r)
    
# Equal aspect ratio Figure with black background and no axes.
fig, ax = plt.subplots(1, 1, figsize=(14,8))
ax.grid()
line1, = ax.plot(x, p1[0], lw=3, label='Species 1, $d_1=${:.3f}'.format(d1))
line2, = ax.plot(x, p2[0], lw=3, label='Species 2, $d_2=${:.3f}'.format(d2))
line3, = ax.plot(x, r, ls='--', lw=3)
ax.legend(fontsize=20)

ani = animation.FuncAnimation(fig, animate, frames=400, interval=50)
ani.save('species.gif',writer='pillow',fps=10)

"""
class SpatialSIZR:
    def __init__(self, sigma, beta, rho, alpha, delta_S, delta_I, delta_Z, S0, I0, Z0, R0, dS, dI, dZ, aS, aI, r):
        for name, argument in locals().items():
            if name not in ('self', 'S0', 'I0', 'R0', 'Z0'):
                if isinstance(argument, (float, int)):
                    setattr(self, name, lambda self, value=argument: value)
                elif callable(argument):
                    setattr(self, name, argument)

        self.initial_conditions = [S0, I0, Z0, R0]

    def __call__(self, u, t, x):
        # RHS of system of PDEs

        S, I, Z, R = u

        dSdt = self.sigma(t) - self.beta(t) * S * Z - self.delta_S(t) * S - self.alpha(t) * S ** 2 + self.dS(t) * (S[x-1] - 2 * S[x] + S[x+1]) + self.r(t) * S * (1 - self.aS(t) * Z)
        dIdt = self.beta(t) * S * Z - self.rho(t) * I - self.delta_I(t) * I - self.alpha(t) * I ** 2 + self.dI(t) * (I[x-1] - 2 * I[x] + I[x+1]) + self.r(t) * I * (1 - self.aI(t) * Z)
        dZdt = self.rho(t) * I - self.delta_Z(t) * S * Z - self.alpha(t) * Z ** 2 + self.dZ(t) * (Z[x-1] - 2 * Z[x] + Z[x+1])
        dRdt = self.delta_S(t) * S + self.delta_I(t) * I + self.delta_Z(t) * S * Z + self.alpha(t) * (S ** 2 + I ** 2 + Z ** 2)

        assert abs(dSdt + dIdt + dZdt + dRdt - self.sigma(t)) < 1e-10, "The sum of the derivatives is not zero"

        return [dSdt, dIdt, dZdt, dRdt]


def spatial_sizr_finite_difference(zombie_model, initial_conditions, time_steps, x_steps):
    S0, I0, Z0, R0 = initial_conditions
    dt = time_steps[1] - time_steps[0]
    dx = x_steps[1] - x_steps[0]
    S = np.zeros((len(time_steps), len(x_steps)))
    I = np.zeros((len(time_steps), len(x_steps)))
    Z = np.zeros((len(time_steps), len(x_steps)))
    R = np.zeros((len(time_steps), len(x_steps)))

    S[0, :] = S0
    I[0, :] = I0
    Z[0, :] = Z0
    R[0, :] = R0

    for t in range(1, len(time_steps)):
        for x in range(1, len(x_steps) - 1):
            dSdt, dIdt, dZdt, dRdt = zombie_model([S[t - 1, x], I[t - 1, x], Z[t - 1, x], R[t - 1, x]], time_steps[t], x)
            S[t, x] = S[t - 1, x] + dt * dSdt
            I[t, x] = I[t - 1, x] + dt * dIdt
            Z[t, x] = Z[t - 1, x] + dt * dZdt
            R[t, x] = R[t - 1, x] + dt * dRdt

        # Apply zero-flux boundary conditions
        S[t, 0], S[t, -1] = S[t, 1], S[t, -2]
        I[t, 0], I[t, -1] = I[t, 1], I[t, -2]
        Z[t, 0], Z[t, -1] = Z[t, 1], Z[t, -2]
        R[t, 0], R[t, -1] = R[t, 1], R[t, -2]

    return S, I, Z, R

# Example usage
time_steps = np.linspace(0, 10, 100)
x_steps = np.linspace(0, 1, 100)
S0 = np.ones(len(x_steps))
I0 = np.zeros(len(x_steps))
Z0 = np.zeros(len(x_steps))
R0 = np.zeros(len(x_steps))

# Instantiate the SpatialSIZR model with example parameters
zombie_model = SpatialSIZR(sigma=0.1, beta=0.2, rho=0.1, alpha=0.1, delta_S=0.1, delta_I=0.1, delta_Z=0.1, S0=S0, I0=I0, Z0=Z0, R0=R0, dS=0.01, dI=0.01, dZ=0.01, aS=0.5, aI=0.5, r=1)

# Run the simulation
S, I, Z, R = spatial_sizr_finite_difference(zombie_model, [S0, I0, Z0, R0], time_steps, x_steps)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(x_steps, S[-1, :], label='S')
plt.plot(x_steps, I[-1, :], label='I')
plt.plot(x_steps, Z[-1, :], label='Z')
plt.plot(x_steps, R[-1, :], label='R')
plt.legend()
plt.show()
"""
