
# SIR model

# with: people with disease, to decide the infection rate and kill rate
# without: extreme case of disease, underutilize location info, natural death vs turned into zombie two rate, seir model, other epidemic model example

# https://mysite.science.uottawa.ca/rsmith43/Zombies.pdf
# https://www.youtube.com/watch?v=AvzQ-F3W708


# without: Lotka–Volterra equations

# https://www.youtube.com/watch?v=qwrp3lB-jkQ
# https://gautamdayal.github.io/natural-selection/



import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


class SIZR:
    def __init__(self, params):
        """
        SIZR model for simulating zombie outbreaks.
        
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
        self.params = params
        self.initial_conditions = [params['S0'], params['I0'], params['Z0'], params['R0']]

    def _evaluate_param(self, param_name, t, S, I, Z, R):
        """Evaluate parameters, supporting both constants and functions."""
        param = self.params[param_name]
        return param(t, S, I, Z, R) if callable(param) else param

    def __call__(self, u, t):
        """Compute derivatives for the ODE system."""
        S, I, Z, R = u

        sigma = self._evaluate_param('sigma', t, S, I, Z, R)
        beta = self._evaluate_param('beta', t, S, I, Z, R)
        rho = self._evaluate_param('rho', t, S, I, Z, R)
        alpha = self._evaluate_param('alpha', t, S, I, Z, R)
        delta_S = self._evaluate_param('delta_S', t, S, I, Z, R)
        delta_I = self._evaluate_param('delta_I', t, S, I, Z, R)
        delta_Z = self._evaluate_param('delta_Z', t, S, I, Z, R)

        dSdt = sigma - beta * S * Z - delta_S * S - alpha * S**2
        dIdt = beta * S * Z - rho * I - delta_I * I - alpha * I**2
        dZdt = rho * I - delta_Z * Z - alpha * Z**2
        dRdt = delta_S * S + delta_I * I + delta_Z * Z + alpha * (S**2 + I**2 + Z**2)
        
        assert np.isclose(dSdt + dIdt + dZdt + dRdt, sigma), "Total population should be constant"

        return [dSdt, dIdt, dZdt, dRdt]

    def plot(self, time_steps, solution):
        """Plot the results of the SIZR model."""
        H, I, Z, D = solution.T
        plt.figure(figsize=(12, 8))
        plt.plot(time_steps, H, label='Humans', color='blue', linestyle='-')
        plt.plot(time_steps, I, label='Infected', color='orange', linestyle='--')
        plt.plot(time_steps, Z, label='Zombies', color='red', linestyle='-.')
        plt.plot(time_steps, D, label='Dead', color='green', linestyle=':')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        
        title = (f"Zombie Apocalypse Simulation\n"
                f"σ={self.params['sigma']}, β={self.params['beta']}, ρ={self.params['rho']}, α={self.params['alpha']}\n"
                f"δS={self.params['delta_S']}, δI={self.params['delta_I']}, δZ={self.params['delta_Z']}\n"
                f"S0={self.params['S0']}, I0={self.params['I0']}, Z0={self.params['Z0']}, R0={self.params['R0']}")
        plt.title(title)
        
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage

# params = {
#     'sigma': lambda t, S, I, Z, R: 2 * np.exp(-Z / (0.1 * (S + I + Z + R))),
#     'beta': lambda t, S, I, Z, R: 0.03 * np.exp(-Z / (0.1 * (S + I + Z + R))),
#     'rho': lambda t, S, I, Z, R: 1 * np.exp(-I / (0.05 * (S + I + Z + R))),
#     'delta_Z': lambda t, S, I, Z, R: 0.0016 + 0.0434 * np.tanh(((S + I + Z + R) - S) / (0.5 * (S + I + Z + R))),
#     'delta_S': lambda t, S, I, Z, R: 0.007 * np.exp(-Z / (0.2 * (S + I + Z + R))),
#     'delta_I': lambda t, S, I, Z, R: 0.014 * np.exp(-Z / (0.2 * (S + I + Z + R))),
#     'alpha': lambda t, S, I, Z, R: 0.0001 * np.exp((S + I + Z + R) / 1000),
#     'S0': 100, 'I0': 0, 'Z0': 10, 'R0': 0
# }

"""
The three phases of the Zombie apocaplypse
Phase 1: initial phase
Lasts four hours. Some humans meet one zombie.

Phase 2: Hysteria
Lasts 24 hours. Zombie threat is evident.

Phase 3: Counter-attack
Lasts five hours.
"""
# params = {
#     'sigma': lambda t, S, I, Z, R: 20 if t < 4 else (2 if 4 <= t < 28 else 0),
#     'beta': lambda t, S, I, Z, R: 0.03 if t < 4 else (0.0012 if 4 <= t < 28 else 0),
#     'rho': 1,
#     'alpha': 0.0001,
#     'delta_S': lambda t, S, I, Z, R: 0 if t < 28 else 0.007,
#     'delta_I': lambda t, S, I, Z, R: 0 if t < 4 else (0.014 if 4 <= t < 28 else 0.05),
#     'delta_Z': lambda t, S, I, Z, R: 0 if t < 4 else (0.0016 if 4 <= t < 28 else 0.05),
#     'S0': 100, 'I0': 0, 'Z0': 10, 'R0': 0
# }

params = {
    'sigma': 2,  # Constant rate of population birth
    'beta': 0.012,  # Constant rate of transmission between susceptible and zombie populations
    'rho': 1,  # Constant rate of transformation from infected to zombie
    'alpha': 0.0001,  # Constant overpopulation factor
    'delta_S': 0.0,  # Constant death rate of susceptible individuals
    'delta_I': 0.014,  # Constant death rate of infected individuals
    'delta_Z': 0.0016,  # Constant death rate of zombies
    'S0': 100,  # Initial susceptible population
    'I0': 0,  # Initial infected population
    'Z0': 10,  # Initial zombie population
    'R0': 0  # Initial removed (dead or immune) population
}

zombie_model = SIZR(params)
time_steps = np.linspace(0, 33, 10000)
solution = odeint(zombie_model, zombie_model.initial_conditions, time_steps)
zombie_model.plot(time_steps, solution)


"""
Conditional Switching based on specific conditions or thresholds to represent discrete events or interventions.

Incorporate stochastic element , adding a probabilistic aspect to the model. This can be particularly useful in capturing the unpredictable nature of a zombie outbreak.
"""
