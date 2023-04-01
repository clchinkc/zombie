
# https://github.com/gregwinther/youtube/tree/master/ODESolver

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from tqdm import tqdm


class ODESolver:
    """ODESolver
    Solves ODE on the form:
    u' = f(u, t), u(0) = U0
    Parameters
    ----------
    f : callable
        Right-hand-side function f(u, t)
    """

    def __init__(self, f):
        self.f = f

    def set_initial_conditions(self, U0):
        """
        Sets initial conditions
        Parameters
        ----------
        U0 : int or float or array_like
            Inital condition(s)
        """
        if isinstance(U0, (int, float)):
            # Scalar ODE
            self.number_of_eqns = 1
            U0 = float(U0)
        else:
            # System of eqns
            U0 = np.asarray(U0)
            self.number_of_eqns = U0.size
        self.U0 = U0

    def solve(self, time_points):
        """
        Solves ODE according to given time points.
        The resolution is implied by spacing of 
        time points.
        Parameters
        ----------
        time_points : array_like
            Time points to solve for
        
        Returns
        -------
        u : array_like
            Solution
        t : array_like
            Time points corresponding to solution
        """

        self.t = np.asarray(time_points)
        n = self.t.size

        self.u = np.zeros((n, self.number_of_eqns))

        self.u[0, :] = self.U0

        # Integrate
        for i in tqdm(range(n - 1), ascii=True):
            self.i = i
            self.u[i + 1] = self.advance()

        return self.u, self.t
        
    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError

# Poor performance
class ForwardEuler(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        return u[i, :] + dt * f(u[i, :], t[i])

class RungeKutta4(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        dt2 = dt / 2
        K1 = dt * f(u[i, :], t[i])
        K2 = dt * f(u[i, :] + 0.5 * K1, t[i] + dt2)
        K3 = dt * f(u[i, :] + 0.5 * K2, t[i] + dt2)
        K4 = dt * f(u[i, :] + K3, t[i] + dt)
        return u[i, :] + (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

class Heun(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        K1 = dt * f(u[i, :], t[i])
        K2 = dt * f(u[i, :] + K1, t[i] + dt)
        return u[i, :] + 0.5 * (K1 + K2)

# Poor performance
class BackwardEuler(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        u_next_guess = u[i, :] + dt * f(u[i, :], t[i])
        residual = lambda u_next: u_next - u[i, :] - dt * f(u_next, t[i + 1])
        u_next = fsolve(residual, u_next_guess)
        return u_next

class AdamsBashforth(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        if i == 0:
            K1 = dt * f(u[i, :], t[i])
            u_next = u[i, :] + K1
        else:
            K1 = dt * f(u[i, :], t[i])
            K2 = dt * f(u[i-1, :], t[i-1])
            u_next = u[i, :] + (3/2)*K1 - (1/2)*K2
        return u_next

class AdamsMoulton(ODESolver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i + 1] - t[i]
        if i == 0:
            K1 = dt * f(u[i, :], t[i])
            u_next_guess = u[i, :] + K1
            residual = lambda u_next: u_next - u[i, :] - (dt/2) * (f(u_next, t[i+1]) + f(u[i, :], t[i]))
            u_next = fsolve(residual, u_next_guess)
        else:
            K1 = dt * f(u[i, :], t[i])
            K2 = dt * f(u[i-1, :], t[i-1])
            u_next_guess = u[i, :] + (5/12)*K1 + (8/12)*K2 - (1/12)*dt*f(u[i-2, :], t[i-2])
            residual = lambda u_next: u_next - u[i, :] - (dt/12) * (5*f(u_next, t[i+1]) + 8*f(u[i, :], t[i]) - f(u[i-1, :], t[i-1]))
            u_next = fsolve(residual, u_next_guess)
        return u_next

if __name__ == '__main__':
    # Define difficult and time-consuming ODE
    def f(u, t):
        return np.exp(-t) * np.sin(3 * np.pi * t) + np.exp(-t) * np.cos(5 * np.pi * t)

    # Define time points
    t = np.linspace(0, 10, 1000)

    # Define solvers
    solvers = [
        ForwardEuler(f),
        RungeKutta4(f),
        Heun(f),
        BackwardEuler(f),
        AdamsBashforth(f),
        AdamsMoulton(f),
    ]

    # Define initial condition
    U0 = 1

    # Solve ODE
    for solver in solvers:
        solver.set_initial_conditions(U0)
        u, t = solver.solve(t)
        plt.plot(t, u, label=solver.__class__.__name__)
        
    # Plot exact solution
    u_exact = odeint(f, U0, t)
    plt.plot(t, u_exact, label='Exact')

    plt.legend()
    plt.show()

# crank nicolson method
# finite difference method
