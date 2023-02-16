
# SIR model

"""
The SIR model is a mathematical framework used to describe the spread of infectious diseases. It divides a population into three groups: susceptible, infected, and recovered. The susceptible group consists of individuals who are not yet infected but are at risk of becoming infected. The infected group consists of individuals who have contracted the disease and are capable of transmitting it to others. The recovered group consists of individuals who have recovered from the disease and are no longer infectious.
In the context of a zombie apocalypse simulation, the SIR model can help to estimate the number of individuals who are susceptible, infected, and recovered at any given time. The model can be adapted to incorporate the unique characteristics of a zombie apocalypse, such as the transmission mechanisms, the speed of infection, and the rate of recovery or death, to determine the progression of the disease and its impact on the population.
By modeling the spread of the zombie virus, the SIR model can help to estimate the number of survivors at any given time, which can inform the development of evacuation and containment plans. It can also help to determine the rate at which the zombie population is growing, which can be useful in planning for military interventions and resource allocation.
Furthermore, the SIR model can be used to evaluate the impact of different response strategies, such as the implementation of quarantine measures, the use of barriers to block the spread of the virus, or the deployment of military forces to contain outbreaks. By simulating different scenarios, the model can help to determine the most effective response strategies and allocate resources accordingly.
In conclusion, the SIR model can provide valuable insights into the spread and impact of a zombie apocalypse, which can inform effective response strategies and mitigate the consequences of the outbreak. By modeling the progression of the disease and its impact on a population, the SIR model can play a critical role in zombie apocalypse simulations.
"""

"""
To design a SIR model simulation of human and zombie populations in a zombie apocalypse, we will start with the following assumptions:
Susceptible individuals: The susceptible group consists of individuals who are not yet infected with the zombie virus but are at risk of becoming infected.
Infected individuals: The infected group consists of individuals who have been bitten by a zombie and are in the process of turning into zombies.
Recovered individuals: The recovered group consists of individuals who have died as a result of the zombie virus and have become zombies.

We will also assume the following parameters:
Transmission rate: The rate at which the zombie virus spreads from one individual to another.
Incubation period: The time it takes for an infected individual to turn into a zombie after being bitten.
Death rate: The rate at which infected individuals die as a result of the zombie virus.

Given these assumptions and parameters, we can now write the following differential equations to model the progression of the human and zombie populations:
dS/dt = -(beta * S * Z)/(S + I + R)
dI/dt = (beta * S * Z)/(S + I + R) - (gamma + delta) * I
dR/dt = gamma * I
dZ/dt = delta * I

Where:
S = number of susceptible individuals
I = number of infected individuals
R = number of recovered individuals
Z = number of zombies
beta = transmission rate
gamma = incubation period
delta = death rate

We can then use numerical methods, such as the Euler method, to solve the differential equations and estimate the progression of the human and zombie populations over time. This simulation can then be used to evaluate the impact of different response strategies and allocate resources accordingly.

In conclusion, the SIR model provides a useful framework for modeling the spread and impact of a zombie apocalypse and can inform effective response strategies to mitigate the consequences of the outbreak.
"""

# with: people with disease, to decide the infection rate and kill rate
# without: extreme case of disease, underutilize location info, natural death vs turned into zombie two rate, seir model, other epidemic model example

# https://mysite.science.uottawa.ca/rsmith43/Zombies.pdf
# https://www.youtube.com/watch?v=AvzQ-F3W708

# https://github.com/gregwinther/youtube/tree/master/ODESolver
# https://github.com/gregwinther/youtube/blob/master/disease_simulations/sizr.py

# without: Lotka–Volterra equations

# https://www.youtube.com/watch?v=qwrp3lB-jkQ
# https://gautamdayal.github.io/natural-selection/

from time import perf_counter

import numba

from ODESolver import ForwardEuler

"""
The Zombie Apocaplypse!
S' = sigma - beta*S*Z - delta_S*S
I' = beta*S*Z - rho*I - delta_I*I
Z' = rho*I - alpha*S*Z
R' = delta_S*S + delta_I*I + alpha*S*Z
"""

import numpy as np
from matplotlib import pyplot as plt


class SIZR:
    def __init__(
        self, sigma, beta, rho, delta_S, delta_I, alpha, S0, I0, Z0, R0
    ):
        """
        The Zombie class
        initial value:  S0, I0, Z0, R0
        """

        for name, argument in locals().items():
            if name not in ('self', 'S0', 'I0', 'R0', 'Z0'):
                if isinstance(argument, (float, int)):
                    setattr(self, name, lambda self, value=argument: value)
                elif callable(argument):
                    setattr(self, name, argument)

        self.initial_conditions = [S0, I0, Z0, R0]

    def __call__(self, u, t):
        """RHS of system of ODEs"""

        S, I, Z, _ = u 

        return np.asarray([
            self.sigma(t) - self.beta(t)*S*Z - self.delta_S(t)*S,
            self.beta(t)*S*Z - self.rho(t)*I - self.delta_I(t)*I,
            self.rho(t)*I - self.alpha(t)*S*Z,
            self.delta_S(t)*S + self.delta_I(t)*I + self.alpha(t)*I 
        ])

if __name__ == "__main__":

    """ The three phases of the Zombie apocaplypse
    Phase 1: initial phase
    Lasts four hours. Some humans meet one zombie.
    sigma = 20, beta = 0.03, rho = 1, S0 = 60, Z0 = 1
    Phase 2: Hysteria
    Lasts 24 hours. Zombie threat is evident.
    beta = 0.0012, alpha = 0.0016, delta_I = 0.014, sigma = 2,
    rho = 1
    Phase 3: Counter-attack
    Lasts five hours.
    alpha = 0.006, beta = 0 (no humans are infected),
    delta_S = 0.007, rho = 1, delta_I = 0.05
    """

    beta = lambda t: 0.03 if t < 4 else (0.0012 if t > 4 and t < 28 else 0)
    alpha = lambda t: 0 if t < 4 else (0.0016 if t > 4 and t < 28 else 0.05)
    sigma = lambda t: 20 if t < 4 else (2 if t > 4 and t < 28  else 0)
    rho = 1
    delta_I = lambda t: 0 if t < 4 else (0.014 if t > 4 and t < 28 else 0.05)
    delta_S = lambda t: 0 if t < 28 else 0.007

    # beta = 0.012
    # alpha = 0.0016
    # sigma = 2
    # rho = 1
    # delta_I = 0.014
    # delta_S = 0.0

    S0 = 60
    I0 = 0
    Z0 = 1
    R0 = 0

    zombie_model = SIZR(
        sigma, beta, rho, delta_S, delta_I, alpha, S0, I0, Z0, R0
    )
    solver = ForwardEuler(zombie_model)
    solver.set_initial_conditions(zombie_model.initial_conditions)
    
    time_steps = np.linspace(0, 33, 10001)
    
    t1_start = perf_counter()
    for i in range(10):
        u, t = solver.solve(time_steps)
    t1_stop = perf_counter()
    print("Elapsed time:", t1_stop - t1_start)

    plt.plot(t, u[:, 0], label="Susceptible humans")
    plt.plot(t, u[:, 1], label="Infected humans")
    plt.plot(t, u[:, 2], label="Zombies")
    plt.plot(t, u[:, 3], label="Dead")
    plt.legend()
    plt.show()
    

