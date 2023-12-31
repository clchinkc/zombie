import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fft2, fftshift

# Simulation parameters
grid_size = 50  # Size of the grid
initial_infected = 5  # Initial number of infected cells
infection_rate = 0.2  # Probability of infection spread to adjacent cells
simulation_steps = 100  # Number of steps in the simulation

# Initialize the grid (0: Susceptible, 1: Infected, 2: Removed)
grid = np.zeros((grid_size, grid_size), dtype=int)
infected_indices = np.random.choice(grid_size * grid_size, initial_infected, replace=False)
grid[np.unravel_index(infected_indices, grid.shape)] = 1

# Function to update the grid based on zombie spread dynamics
def update_grid(grid):
    new_grid = grid.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:  # Infected cell
                # Check adjacent cells for infection spread
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if 0 <= i + di < grid_size and 0 <= j + dj < grid_size:
                            if grid[i + di, j + dj] == 0:  # Susceptible cell
                                if np.random.rand() < infection_rate:
                                    new_grid[i + di, j + dj] = 1  # New infection
    return new_grid

# Run the simulation
time_series_data = []
spatial_data = []

for step in range(simulation_steps):
    grid = update_grid(grid)
    time_series_data.append(np.sum(grid == 1))  # Count infected cells
    spatial_data.append(grid.copy())

# Convert spatial data to array for easier processing
spatial_data = np.array(spatial_data)

# FFT analysis on time-series data
time_series_fft = fft(time_series_data)
frequencies = np.fft.fftfreq(len(time_series_data), d=1)

# Identify and highlight dominant frequencies
dominant_freqs = frequencies[(np.abs(time_series_fft) > np.max(np.abs(time_series_fft)) * 0.1) & (frequencies > 0)]
dominant_freq_indices = np.argsort(-np.abs(time_series_fft))[:len(dominant_freqs)]

# Spatial analysis at different stages
stages = [10, 30, 70]  # Early, middle, and late stages of the simulation

# Visualization
fig, axs = plt.subplots(len(stages) + 1, 2, figsize=(12, (len(stages) + 1) * 8), constrained_layout=True)

# Plotting time-series data and its FFT
axs[0, 0].plot(frequencies, np.abs(time_series_fft))
axs[0, 0].scatter(frequencies[dominant_freq_indices], np.abs(time_series_fft)[dominant_freq_indices], color='red')
axs[0, 0].set_title("FFT of Time-Series Data (Dominant Frequencies Highlighted)")
axs[0, 0].set_xlabel("Frequency")
axs[0, 0].set_ylabel("Amplitude")

axs[0, 1].plot(time_series_data)
axs[0, 1].set_title("Time-Series Data")
axs[0, 1].set_xlabel("Time Step")
axs[0, 1].set_ylabel("Number of Infected Cells")

# Plotting spatial data and its FFT at different stages
for i, stage in enumerate(stages):
    # FFT of spatial data at the selected stage
    spatial_fft = fft2(spatial_data[stage])
    spatial_fft_shifted = fftshift(spatial_fft)

    # Original spatial data
    axs[i + 1, 0].imshow(spatial_data[stage], cmap='viridis')
    axs[i + 1, 0].set_title(f"Spatial Data at Time Step {stage}")
    axs[i + 1, 0].set_xlabel("X Coordinate")
    axs[i + 1, 0].set_ylabel("Y Coordinate")

    # FFT-transformed spatial data
    axs[i + 1, 1].imshow(np.log(np.abs(spatial_fft_shifted) + 1e-10), cmap='hot')
    axs[i + 1, 1].set_title(f"FFT of Spatial Data at Time Step {stage} (Log Scale)")
    axs[i + 1, 1].set_xlabel("Frequency X")
    axs[i + 1, 1].set_ylabel("Frequency Y")

plt.show()
