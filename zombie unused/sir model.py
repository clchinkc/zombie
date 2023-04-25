
# SIR model

# with: people with disease, to decide the infection rate and kill rate
# without: extreme case of disease, underutilize location info, natural death vs turned into zombie two rate, seir model, other epidemic model example

# https://mysite.science.uottawa.ca/rsmith43/Zombies.pdf
# https://www.youtube.com/watch?v=AvzQ-F3W708


# without: Lotka–Volterra equations

# https://www.youtube.com/watch?v=qwrp3lB-jkQ
# https://gautamdayal.github.io/natural-selection/


"""
The Zombie Apocaplypse!

To design a SIR model simulation of human and zombie populations in a zombie apocalypse, we will start with the following assumptions:
Susceptible individuals: The susceptible group consists of individuals who are not yet infected with the zombie virus but are at risk of becoming infected.
Infected individuals: The infected group consists of individuals who have been bitten by a zombie and are in the process of turning into zombies.
Recovered individuals: The recovered group consists of individuals who have died as a result of the zombie virus and have become zombies.

We will also assume the following parameters:
Transmission rate: The rate at which the zombie virus spreads from one individual to another.
Incubation period: The time it takes for an infected individual to turn into a zombie after being bitten.
Death rate: The rate at which infected individuals die as a result of the zombie virus.

Given these assumptions and parameters, we can now write the following differential equations to model the progression of the human and zombie populations:

S' = sigma - beta*S*Z - delta_S*S - alpha*S**2
I' = beta*S*Z - rho*I - delta_I*I - alpha*I**2
Z' = rho*I - delta_Z*S*Z - alpha*Z*I - alpha*Z**2
R' = delta_S*S + delta_I*I + delta_Z*S*Z + alpha*S**2 + alpha*I**2 + alpha*Z**2

Where:
S = number of susceptible individuals
I = number of infected individuals
Z = number of zombies
R = number of recovered individuals
sigma = birth rate
beta = transmission rate
rho = transformation rate
alpha = overpopulation factor
delta_S = death rate of susceptible individuals
delta_I = death rate of infected individuals
delta_Z = death rate of zombies

We can then use numerical methods, such as the Euler method, to solve the differential equations and estimate the progression of the human and zombie populations over time. This simulation can then be used to evaluate the impact of different response strategies and allocate resources accordingly.
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


class SIZR:
    def __init__(self, sigma, beta, rho, alpha, delta_S, delta_I, delta_Z, S0, I0, Z0, R0):
        """
        The Zombie class
        
        Parameters:
        - sigma: float or callable, rate of population birth
        - beta: float or callable, rate of transmission between susceptible and zombie populations
        - rho: float or callable, rate of recovery from zombie infection
        - alpha: float or callable, rate of decay of zombies due to natural causes or combat
        - delta_S: float or callable, rate of susceptible population death 
        - delta_I: float or callable, rate of infected population death
        - delta_Z: float or callable, rate of zombie population death
        - S0: float, initial susceptible population
        - I0: float, initial infected population
        - Z0: float, initial zombie population
        - R0: float, initial removed (dead or immune) population
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

        S, I, Z, R = u 

        dSdt = self.sigma(t) - self.beta(t)*S*Z - self.delta_S(t)*S - self.alpha(t)*S**2
        dIdt = self.beta(t)*S*Z - self.rho(t)*I - self.delta_I(t)*I - self.alpha(t)*I**2
        dZdt = self.rho(t)*I - self.delta_Z(t)*S*Z - self.alpha(t)*Z**2
        dRdt = self.delta_S(t)*S + self.delta_I(t)*I + self.delta_Z(t)*S*Z + self.alpha(t)*(S**2 + I**2 + Z**2)
        
        assert abs(dSdt + dIdt + dZdt + dRdt - self.sigma(t)) < 1e-10, "The sum of the derivatives is not zero"
        
        return [dSdt, dIdt, dZdt, dRdt]

if __name__ == "__main__":

    """ The three phases of the Zombie apocaplypse
    Phase 1: initial phase
    Lasts four hours. Some humans meet one zombie.

    Phase 2: Hysteria
    Lasts 24 hours. Zombie threat is evident.

    Phase 3: Counter-attack
    Lasts five hours.

    """

    beta = lambda t: 0.03 if t < 4 else (0.0012 if t > 4 and t < 28 else 0)
    delta_Z = lambda t: 0 if t < 4 else (0.0016 if t > 4 and t < 28 else 0.05)
    sigma = lambda t: 20 if t < 4 else (2 if t > 4 and t < 28  else 0)
    rho = 1
    alpha = 0.0001
    delta_I = lambda t: 0 if t < 4 else (0.014 if t > 4 and t < 28 else 0.05)
    delta_S = lambda t: 0 if t < 28 else 0.007

    # sigma = 2 # rate of population birth
    # beta = 0.012 # rate of transmission between susceptible and zombie populations
    # rho = 1 # rate of transformation from infected to zombie
    # alpha = 0.0001 # overpopulation factor
    # delta_S = 0.0 # death rate of susceptible individuals
    # delta_I = 0.014 # death rate of infected individuals
    # delta_Z = 0.0016 # death rate of zombies

    S0 = 100 # initial susceptible population
    I0 = 0 # initial infected population
    Z0 = 10 # initial zombie population
    R0 = 0 # initial removed (dead or immune) population
    
    zombie_model = SIZR(sigma, beta, rho, alpha, delta_S, delta_I, delta_Z, S0, I0, Z0, R0)
    
    time_steps = np.linspace(0, 33, 10000)
    
    sol_HZD = odeint(zombie_model, zombie_model.initial_conditions, time_steps)
    
    H, I, Z, D = sol_HZD.T
    
    plt.plot(time_steps, H, label='Humans')
    plt.plot(time_steps, I, label='Infected')
    plt.plot(time_steps, Z, label='Zombies')
    plt.plot(time_steps, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Zombie Apocaplypse SIZR Model with Overpopulation Factor Simulation')
    plt.legend()
    plt.show()



"""
A zombie apocalypse population model that takes into account overpopulation can be created by combining the classical SIZR (Susceptible, Infected, Zombie, Removed) model with a population model that accounts for overpopulation.

In the SIZR model, the population is divided into four compartments: Susceptible (S), Infected (I), Zombie (Z), and Removed (R). The population dynamics of each compartment are governed by a set of differential equations that describe the rate at which individuals move between the compartments. The equations are as follows:

dS/dt = -βSZ/N
dI/dt = βSZ/N - γI
dZ/dt = γI - αSZ/N
dR/dt = αSZ/N

where β is the infection rate, γ is the recovery rate, α is the rate at which zombies are killed, and N is the total population size.

To account for overpopulation, we can modify the population size N in the above equations to reflect the fact that the population is exceeding its carrying capacity. This can be done by introducing a carrying capacity K, such that the population size is limited to K:

N = K - S - I - Z - R

The modified equations become:

dS/dt = -βSZ/(K-S-I-Z-R)
dI/dt = βSZ/(K-S-I-Z-R) - γI
dZ/dt = γI - αSZ/(K-S-I-Z-R)
dR/dt = αSZ/(K-S-I-Z-R)

In this model, the carrying capacity K represents the maximum number of individuals that the environment can sustainably support. When the population exceeds the carrying capacity, the rate at which individuals move between compartments is reduced, reflecting the fact that there are fewer resources available to support the population. The model assumes that the carrying capacity remains constant over time, which may not be realistic in the case of a zombie apocalypse, as the environment would likely be heavily impacted by the outbreak.

Overall, this model provides a framework for understanding how overpopulation could impact the dynamics of a zombie apocalypse. The model suggests that overpopulation could reduce the effectiveness of interventions aimed at controlling the outbreak, such as quarantine or vaccination, by limiting the resources available to support the population.
"""