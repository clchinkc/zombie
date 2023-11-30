
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
        
        assert np.isclose(dSdt + dIdt + dZdt + dRdt, self.sigma(t)), "The sum of the derivatives is not zero"
        
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
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


class SIZR:
    def __init__(self, S0, I0, Z0, R0, birth_rate_factor, transmission_rate_factor, recovery_rate_factor, zombie_death_base, zombie_death_scale, natural_death_s_base, natural_death_i_base, overpopulation_factor):
        self.S0, self.I0, self.Z0, self.R0 = S0, I0, Z0, R0
        self.birth_rate_factor = birth_rate_factor
        self.transmission_rate_factor = transmission_rate_factor
        self.recovery_rate_factor = recovery_rate_factor
        self.zombie_death_base = zombie_death_base
        self.zombie_death_scale = zombie_death_scale
        self.natural_death_s_base = natural_death_s_base
        self.natural_death_i_base = natural_death_i_base
        self.overpopulation_factor = overpopulation_factor

    # Continuous function for birth rate
    def sigma(self, Z, total_population):
        return self.birth_rate_factor * np.exp(-Z / (0.1 * total_population))

    # Continuous function for transmission rate
    def beta(self, S, Z, total_population):
        return self.transmission_rate_factor * np.exp(-Z / (0.1 * total_population))

    # Continuous function for recovery rate
    def rho(self, I, total_population):
        return self.recovery_rate_factor * np.exp(-I / (0.05 * total_population))

    # Continuous function for zombie death rate
    def delta_Z(self, S, total_population):
        return self.zombie_death_base + self.zombie_death_scale * np.tanh((total_population - S) / (0.5 * total_population))

    # Continuous function for natural death rate of susceptibles
    def delta_S(self, Z, total_population):
        return self.natural_death_s_base * np.exp(-Z / (0.2 * total_population))

    # Continuous function for natural death rate of infected
    def delta_I(self, Z, total_population):
        return self.natural_death_i_base * np.exp(-Z / (0.2 * total_population))

    # Continuous function for overpopulation factor
    def alpha(self, total_population):
        return self.overpopulation_factor * np.exp(total_population / 1000)

    def __call__(self, u, t):
        S, I, Z, R = u 
        total_population = S + I + Z + R

        # Calculate dynamic parameters
        sigma = self.sigma(Z, total_population)
        beta = self.beta(S, Z, total_population)
        delta_S = self.delta_S(Z, total_population)
        delta_I = self.delta_I(Z, total_population)
        rho = self.rho(I, total_population)
        delta_Z = self.delta_Z(S, total_population)
        alpha = self.alpha(total_population)

        # System of differential equations
        dSdt = sigma - beta * S * Z - delta_S * S - alpha * S**2
        dIdt = beta * S * Z - rho * I - delta_I * I - alpha * I**2
        dZdt = rho * I - delta_Z * S * Z - alpha * Z**2
        dRdt = delta_S * S + delta_I * I + delta_Z * S * Z + alpha * (S**2 + I**2 + Z**2)
        
        return [dSdt, dIdt, dZdt, dRdt]

# Parameters
birth_rate_factor = 2
transmission_rate_factor = 0.03
recovery_rate_factor = 1
zombie_death_base = 0.0016
zombie_death_scale = 0.0434
natural_death_s_base = 0.007
natural_death_i_base = 0.014
overpopulation_factor = 0.0001

# Initialize the model with parameters
S0, I0, Z0, R0 = 100, 0, 10, 0
zombie_model = SIZR(S0, I0, Z0, R0, birth_rate_factor, transmission_rate_factor, recovery_rate_factor, zombie_death_base, zombie_death_scale, natural_death_s_base, natural_death_i_base, overpopulation_factor)


# Simulate over a period
time_steps = np.linspace(0, 33, 10000)
solution = odeint(zombie_model, [S0, I0, Z0, R0], time_steps)

# Plotting
H, I, Z, D = solution.T
plt.plot(time_steps, H, label='Humans')
plt.plot(time_steps, I, label='Infected')
plt.plot(time_steps, Z, label='Zombies')
plt.plot(time_steps, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Dynamic SIZR Model Simulation without Lambda Functions')
plt.legend()
plt.show()

"""
