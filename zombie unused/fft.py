import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift

# Simulation parameters
grid_size = 50  # Size of the grid
initial_infected = 5  # Initial number of infected cells
infection_rate = 0.2  # Probability of infection spread to adjacent cells
simulation_steps = 100  # Number of steps in the simulation

# Initialize the grid (0: Susceptible, 1: Infected, 2: Removed)
grid = np.zeros((grid_size, grid_size), dtype=int)
infected_indices = np.random.choice(grid_size * grid_size, initial_infected, replace=False)
grid[np.unravel_index(infected_indices, grid.shape)] = 1

# Function to update the grid based on infection spread dynamics
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
spatial_data = []
for step in range(simulation_steps):
    grid = update_grid(grid)
    spatial_data.append(grid.copy())

# Convert spatial data to array for easier processing
spatial_data = np.array(spatial_data)

# Function to compute FFT of the spatial data
def compute_fft(data):
    fft_data = fft2(data)
    fft_shifted = fftshift(fft_data)
    return np.log(np.abs(fft_shifted) + 1e-10)  # Log scale for better visualization

# Creating animations
def create_animation(data, cmap, title, xlabel, ylabel, fps=5, interval=200):
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(data[0], cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")

    def update(frame):
        img.set_data(data[frame])
        time_text.set_text(f"Time Step: {frame}")
        return img, time_text

    ani = animation.FuncAnimation(fig, update, frames=range(0, simulation_steps, 5), interval=interval, blit=True)
    return ani

# Creating spatial data animation
ani_spatial = create_animation(spatial_data, 'viridis', "Spatial Data Over Time", "X Coordinate", "Y Coordinate")

# Creating FFT animation
fft_data = [compute_fft(frame) for frame in spatial_data]
ani_fft = create_animation(fft_data, 'hot', "FFT of Spatial Data Over Time", "Frequency X", "Frequency Y")

# Display animations
plt.show()
