import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# Lorenz system dynamics
def lorenz_system(t, xyz, sigma, beta, rho):
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Bifurcation analysis
def bifurcation_diagram(sigma, beta, rho_start, rho_end, initial_condition, steps, transient, dt):
    rhos = np.linspace(rho_start, rho_end, steps)
    bifurcation_data = []

    for rho in rhos:
        # Integrate system over a transient time period and discard this data
        sol_transient = solve_ivp(lorenz_system, [0, transient], initial_condition, args=(sigma, beta, rho))

        # Use the last point from the transient as the new initial condition
        new_initial_condition = sol_transient.y[:, -1]

        # Now integrate the system and use this data for the bifurcation diagram
        sol = solve_ivp(lorenz_system, [0, dt * 1000], new_initial_condition, args=(sigma, beta, rho), t_eval=np.linspace(0, dt * 1000, 1000))
        if sol.success:
            bifurcation_data.append(sol.y[:, -1])
        else:
            bifurcation_data.append([np.nan, np.nan, np.nan])  # In case of solver failure

    return np.array(bifurcation_data), rhos

# Estimate Lyapunov exponents
def estimate_lyapunov_exponents(time_series, dt, sigma, beta, rho):
    n = time_series.shape[1]
    d = 3  # Lorenz system is 3-dimensional
    J = np.zeros((d, d, n))
    for i, xyz in enumerate(time_series.T):
        J[:, :, i] = jacobian_lorenz(xyz, sigma, beta, rho)
    
    # We will average the Lyapunov exponents over the whole time series, after discarding transients
    # Assuming the transient period has already been discarded from the time_series
    Q = np.eye(d)
    exponents = np.zeros((d, n))
    for i in range(n):
        JQ = J[:, :, i] @ Q
        Q, R = np.linalg.qr(JQ)  # Orthonormalize the columns of JQ
        exponents[:, i] = np.diag(R)
    
    # The Lyapunov exponents are the time average of the logarithm of the diagonal elements of R
    l_exp = np.mean(np.log(np.abs(exponents)), axis=1)
    return l_exp

# Jacobian of the Lorenz system
def jacobian_lorenz(xyz, sigma, beta, rho):
    x, y, z = xyz
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])

def estimate_lyapunov_exponent(time_series, dt, sigma, beta, rho):
    n = len(time_series)
    d = len(time_series[0])
    lyapunov_exp = np.zeros(d)
    
    for i in range(1, n - 1):
        J = jacobian_lorenz(time_series[i], sigma, beta, rho)
        Q, R = np.linalg.qr(J)  # QR decomposition for orthonormalization
        lyapunov_exp += np.log(np.abs(np.diagonal(R)))
    
    return lyapunov_exp / (n * dt)

# Simulate the Lorenz system to generate a time series
def simulate_lorenz(timesteps, initial_condition, sigma, beta, rho, dt=0.01):
    t_eval = np.linspace(0, dt*timesteps, timesteps)
    sol = solve_ivp(lorenz_system, [0, dt*timesteps], initial_condition, args=(sigma, beta, rho), t_eval=t_eval, dense_output=True)
    return sol.y

# Parameters for simulation
sigma = 10.0
beta = 8/3.0
rho = 28.0
initial_condition = [1.0, 1.0, 1.0]
timesteps = 10000

# Generate time series from the Lorenz system
time_series = simulate_lorenz(timesteps, initial_condition, sigma, beta, rho)

# Calculate the largest Lyapunov exponent from the time series
dt = 0.01
L_exp = estimate_lyapunov_exponent(time_series.T, dt, sigma, beta, rho)
print(f"Estimated largest Lyapunov exponent: {L_exp[0]}")

# Perform a bifurcation analysis on the 'rho' parameter from 0 to 50
rho_start = 0
rho_end = 50
steps = 1000
transient = 50  # Time to allow the system to settle
bifurcation_data, rhos = bifurcation_diagram(sigma, beta, rho_start, rho_end, initial_condition, steps, transient, dt)

# Plotting the bifurcation diagram
plt.figure(figsize=(10, 6))
plt.plot(rhos, bifurcation_data[:, 0], ',k', alpha=0.5)
plt.title('Bifurcation Diagram')
plt.xlabel('Rho')
plt.ylabel('X')
plt.show()

"""
In the realm of stock price predictions, the marriage of bifurcation analysis and Lyapunov exponents offers a compelling conceptual framework to forecast market dynamics. Here's how such an integrated approach might look:

**Bifurcation Analysis for Market Behavior Prediction**:
The first step would be to engage in bifurcation analysis, treating the stock market as a complex dynamical system influenced by a myriad of parameters. You would track key economic indicators such as interest rates, inflation, market sentiment, and trading volume, which are known to exert substantial impact on market behavior. By systematically varying these parameters within a predictive model, you could potentially identify 'bifurcation points.' These are critical values at which the stock market's behavior abruptly shifts—akin to a phase transition from a bull to a bear market or vice versa. This part of the analysis aims to pinpoint thresholds that signal impending periods of stability or high volatility, allowing investors to brace for qualitative shifts in market trends.

**Lyapunov Exponent for Market Sensitivity Analysis**:
Concurrently, one could draw an analogy to Lyapunov exponents to gauge the market's sensitivity to initial conditions, thus assessing its predictability. Given the impossibility of computing a true Lyapunov exponent for the stock market due to its stochastic and complex nature, one could instead turn to statistical methods that serve as proxies. These methods might involve examining the market's volatility or employing other statistical tools to measure the dispersion of stock prices over time. A high degree of sensitivity (akin to a positive Lyapunov exponent) would hint at a market that is highly unpredictable, where small changes can result in vastly divergent price trajectories. Conversely, low sensitivity (akin to a negative Lyapunov exponent) might indicate a more stable and predictable market.

**Integrated Approach**:
By combining bifurcation analysis with Lyapunov exponent analogies, investors and analysts can form a multi-faceted view of the market's dynamism. They can identify points at which incremental parameter shifts might induce radical changes in market behavior (bifurcation) and understand how sensitive the current market state is to fluctuations (Lyapunov analog). This dual approach could be particularly powerful for designing robust trading strategies that are sensitive to both the onset of new market phases and the degree of randomness or determinism in price movements. Such strategies would be tailored to not only anticipate shifts but also manage risks associated with the inherent unpredictability of stock prices.

For example, a combined model may reveal that when a volatility index reaches a certain high level (a bifurcation point), the market transitions into a highly sensitive state where price trajectories begin to diverge rapidly (indicated by a high proxy Lyapunov exponent), thus signaling a phase of increased unpredictability and potential for rapid market shifts.
"""