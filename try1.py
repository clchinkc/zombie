import pickle
import random
import threading
import time
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Entity:
    def __init__(self, position, boundary):
        self.position = np.array(position)
        self.velocity = np.array([0., 0.])
        self.boundary = np.array(boundary)
        self.canvas_object_id = None

    def move(self, dt):
        new_position = self.position + self.velocity * dt
        # Enforce boundary conditions
        self.position = np.clip(new_position, [0, 0], self.boundary)

class Survivor(Entity):
    def __init__(self, position, boundary):
        super().__init__(position, boundary)
        self.health = 100
        self.health_history = [self.health]
        self.position_history = [position.copy()]  # Store position history

    def decrease_health(self, amount):
        self.health = max(self.health - amount, 0)
        self.health_history.append(self.health)  # Update health history

    def evade_zombie(self, zombie, speed, dt):
        evasion_direction = normalize(self.position - zombie.position)
        # Add random noise to the evasion direction
        evasion_direction += np.random.normal(0, 0.1, 2)
        evasion_direction = normalize(evasion_direction)
        self.velocity = evasion_direction * speed

class Zombie(Entity):
    def __init__(self, position, boundary):
        super().__init__(position, boundary)
        self.position_history = [position.copy()]  # Store position history

class PIDController:
    def __init__(self, kp, ki, kd, control_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.control_limit = control_limit
        self.integral = 0
        self.prev_error = None

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt

        if self.prev_error is None:
            self.prev_error = error

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.control_limit is not None:
            control_signal = np.clip(control_signal, -self.control_limit, self.control_limit)

        return control_signal

def distance(entity1, entity2):
    return float(np.linalg.norm(entity1.position - entity2.position))

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def move_towards(target, mover, speed):
    direction = normalize(target.position - mover.position)
    mover.velocity = direction * speed

class Simulation:
    def __init__(self):
        self.boundary = [20, 20]
        self.dt = 0.1
        self.desired_distance = 5
        self.interaction_distance = 1.0
        self.health_decrement = 5
        self.survivor_speed = 0
        self.zombie_speed = 0.5
        self.steps = 0  # Step counter
        self.paused = True
        self.update_condition = threading.Condition() # Condition for pausing and resuming the simulation
        
        self.survivors = []  # List to store multiple survivors
        self.zombies = []    # List to store multiple zombies
        self.setup_entities()
        self.setup_pid_controller()

        # Data for plotting
        self.health_drop_positions = []

    def setup_entities(self):
        # Clear previous entities
        self.survivors.clear()
        self.zombies.clear()
        # Create multiple survivors and zombies
        num_survivors = 3
        num_zombies = 2
        self.survivors = [Survivor([random.uniform(0, self.boundary[0]), random.uniform(0, self.boundary[1])], self.boundary) for _ in range(num_survivors)]
        self.zombies = [Zombie([random.uniform(0, self.boundary[0]), random.uniform(0, self.boundary[1])], self.boundary) for _ in range(num_zombies)]

    def setup_pid_controller(self):
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.05, control_limit=10)

    def update_simulation(self):
        # Update for multiple entities
        for survivor in self.survivors:
            closest_zombie = min(self.zombies, key=lambda z: distance(survivor, z))
            survivor.evade_zombie(closest_zombie, self.survivor_speed, self.dt)
            survivor.move(self.dt)
            if distance(survivor, closest_zombie) < self.interaction_distance:
                survivor.decrease_health(self.health_decrement)
                self.health_drop_positions.append(survivor.position.copy())
            survivor.health_history.append(survivor.health)  # Record health data

        for zombie in self.zombies:
            closest_survivor = min(self.survivors, key=lambda s: distance(s, zombie))
            move_towards(closest_survivor, zombie, self.zombie_speed)
            zombie.move(self.dt)

        self.steps += 1  # Increment step counter
        self.update_plot_data()

    def update_plot_data(self):
        # Update plot data for multiple entities
        for survivor in self.survivors:
            survivor.position_history.append(survivor.position.copy())
        for zombie in self.zombies:
            zombie.position_history.append(zombie.position.copy())

    def run(self):
        while not self.paused:
            with self.update_condition:
                self.update_simulation()
                self.update_condition.wait()  # Wait for signal from GUI thread

    def start(self):
        if not hasattr(self, 'thread') or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True
            self.thread.start()

    def pause(self):
        self.paused = True
        with self.update_condition:
            self.update_condition.notify_all()

    def resume(self):
        self.paused = False
        self.run()

    def reset(self):
        self.setup_entities()
        self.setup_pid_controller()
        self.health_drop_positions.clear()
        self.steps = 0

    def step_forward(self):
        if self.paused:
            self.update_simulation()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the non-pickleable entries
        del state['update_condition']
        del state['thread']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the non-pickleable entries
        self.update_condition = threading.Condition()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True

    def get_state(self):
        return pickle.dumps(self)

    def set_state(self, state_data):
        state = pickle.loads(state_data)
        self.__setstate__(state.__dict__)

class GUI:
    def __init__(self, simulation):
        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("Survivor vs Zombie Simulation")

        self.setup_control_panel()
        self.setup_canvas()
        self.setup_plot()
        self.setup_performance_metrics()
        self.setup_refresh()
        
        self.debounce_job = None

    def setup_control_panel(self):
        self.control_panel = tk.Frame(self.root)

        self.start_button = tk.Button(self.control_panel, text="Start", command=self.start_simulation)
        self.pause_button = tk.Button(self.control_panel, text="Pause", command=self.pause_simulation)
        self.reset_button = tk.Button(self.control_panel, text="Reset", command=self.reset_simulation)
        self.step_button = tk.Button(self.control_panel, text="Step", command=self.step_simulation)
        self.save_button = tk.Button(self.control_panel, text="Save", command=self.save_simulation)
        self.load_button = tk.Button(self.control_panel, text="Load", command=self.load_simulation)

        self.start_button.pack(side=tk.LEFT)
        self.pause_button.pack(side=tk.LEFT)
        self.reset_button.pack(side=tk.LEFT)
        self.step_button.pack(side=tk.LEFT)
        self.save_button.pack(side=tk.LEFT)
        self.load_button.pack(side=tk.LEFT)

        # Sliders for interactive controls
        self.speed_slider = ttk.Scale(self.control_panel, from_=0, to=10, orient="horizontal", command=self.update_speed)
        self.speed_slider.pack(side=tk.LEFT)
        self.speed_label = tk.Label(self.control_panel, text="Speed: 0")
        self.speed_label.pack(side=tk.LEFT)

        self.control_panel.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    def update_speed(self, event):
        if self.debounce_job is not None:
            self.root.after_cancel(self.debounce_job)
        self.debounce_job = self.root.after(100, self.update_speed_debounced)

    def update_speed_debounced(self):
        speed = self.speed_slider.get()
        self.speed_label.config(text=f"Speed: {speed:.2f}")
        self.simulation.survivor_speed = speed

    def start_simulation(self):
        self.simulation.paused = False
        if not hasattr(self.simulation, 'thread') or not self.simulation.thread.is_alive():
            self.simulation.start()
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)

    def pause_simulation(self):
        self.simulation.pause()
        self.update_canvas()
        self.update_plot()
        self.update_performance_metrics()
        self.pause_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

    def reset_simulation(self):
        self.simulation.reset()
        self.canvas.delete("all")
        self.update_canvas()
        self.update_plot()
        self.update_performance_metrics()

    def step_simulation(self):
        self.simulation.step_forward()
        self.update_canvas()
        self.update_plot()
        self.update_performance_metrics()

    def save_simulation(self):
        with open("simulation_state.pkl", "wb") as file:
            file.write(self.simulation.get_state())

    def load_simulation(self):
        with open("simulation_state.pkl", "rb") as file:
            self.simulation.set_state(file.read())
            self.update_canvas()
            self.update_plot()
            self.update_performance_metrics()

    def setup_canvas(self):
        # Adjust canvas size and positioning to maximize space usage
        self.canvas = tk.Canvas(self.root, width=self.simulation.boundary[0]*10, height=self.simulation.boundary[1]*10)
        self.canvas.pack(side=tk.LEFT, fill=tk.X, expand=False)

    def setup_plot(self):
        # Adjust plot size and positioning
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_fig_widget = self.canvas_fig.get_tk_widget()
        self.canvas_fig_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def setup_performance_metrics(self):
        # Create labels or text areas for performance metrics
        self.performance_frame = tk.Frame(self.root)
        self.survival_time_label = tk.Label(self.performance_frame, text="Survival Time: 0")
        self.distance_label = tk.Label(self.performance_frame, text="Distance Between: 0")
        self.health_label = tk.Label(self.performance_frame, text="Survivor Health: 100")
        self.count_label = tk.Label(self.performance_frame, text="Survivors: 0 | Zombies: 0")
        self.health_depletion_label = tk.Label(self.performance_frame, text="Health Depletion Rate: 0.00")

        self.survival_time_label.pack(side=tk.TOP)
        self.distance_label.pack(side=tk.TOP)
        self.health_label.pack(side=tk.TOP)
        self.count_label.pack(side=tk.TOP)
        self.health_depletion_label.pack(side=tk.TOP)
        self.performance_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

    def update_canvas(self):
        # Update for multiple survivors and zombies
        for survivor in self.simulation.survivors:
            self.update_entity_position(survivor, "green")
        for zombie in self.simulation.zombies:
            self.update_entity_position(zombie, "red")

    def update_entity_position(self, entity, color):
        canvas_height = self.simulation.boundary[1] * 10
        if entity.canvas_object_id is None:
            # If the entity doesn't have a canvas object yet, create one
            entity.canvas_object_id = self.canvas.create_oval(
                entity.position[0] * 10 - 5, canvas_height - entity.position[1] * 10 - 5,
                entity.position[0] * 10 + 5, canvas_height - entity.position[1] * 10 + 5,
                fill=color
            )
        else:
            # If the entity already has a canvas object, just update its position
            self.canvas.coords(
                entity.canvas_object_id,
                entity.position[0] * 10 - 5, canvas_height - entity.position[1] * 10 - 5,
                entity.position[0] * 10 + 5, canvas_height - entity.position[1] * 10 + 5
            )

    def update_position_plot(self):
        self.ax1.clear()

        # Plot positions and add vector field for movement directions
        survivor_positions = np.array([s.position for s in self.simulation.survivors])
        zombie_positions = np.array([z.position for z in self.simulation.zombies])
        survivor_vectors = np.array([s.velocity for s in self.simulation.survivors])
        zombie_vectors = np.array([z.velocity for z in self.simulation.zombies])

        self.ax1.hexbin(survivor_positions[:, 0], survivor_positions[:, 1], gridsize=100, alpha=0.7, cmap='Blues', label='Survivors')
        self.ax1.hexbin(zombie_positions[:, 0], zombie_positions[:, 1], gridsize=100, alpha=0.7, cmap='Reds', label='Zombies')

        # Plot current position of multiple survivors and zombies
        self.ax1.scatter(*zip(*survivor_positions), color="blue", label="Survivors")
        self.ax1.scatter(*zip(*zombie_positions), color="red", label="Zombies")

        # Plot movement directions of multiple survivors and zombies
        self.ax1.quiver(survivor_positions[:, 0], survivor_positions[:, 1], survivor_vectors[:, 0], survivor_vectors[:, 1], color='blue', scale=10, alpha=0.7, label='Survivors')
        self.ax1.quiver(zombie_positions[:, 0], zombie_positions[:, 1], zombie_vectors[:, 0], zombie_vectors[:, 1], color='red', scale=10, alpha=0.7, label='Zombies')

        # Plot position history for multiple survivors and zombies
        for i, survivor in enumerate(self.simulation.survivors):
            self.ax1.plot(*zip(*survivor.position_history), alpha=0.5, color="blue", label="Survivors History" if i == 0 else "")
        for i, zombie in enumerate(self.simulation.zombies):
            self.ax1.plot(*zip(*zombie.position_history), alpha=0.5, color="red", label="Zombies History" if i == 0 else "")

        # Plot health drop positions
        if self.simulation.health_drop_positions:
            self.ax1.scatter(*zip(*self.simulation.health_drop_positions), color="purple", marker='x', label="Health Drop")

        self.ax1.set_xlabel('X Position', fontsize=14)
        self.ax1.set_ylabel('Y Position', fontsize=14)
        self.ax1.set_title('Survivor vs Zombie Position and Movement', fontsize=14, fontweight='bold')
        self.ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax1.legend(loc='upper right')

    def update_distribution_plot(self):
        self.ax2.clear()

        # Define the grid size for the heatmap
        xedges = np.linspace(0, self.simulation.boundary[0], 50)
        yedges = np.linspace(0, self.simulation.boundary[1], 50)

        if len(self.simulation.survivors) > 0:
            survivor_positions = np.array([s.position for s in self.simulation.survivors])
            H, _, _ = np.histogram2d(survivor_positions[:, 0], survivor_positions[:, 1], bins=[xedges, yedges])
            self.ax2.imshow(H.T, origin='lower', extent=[0, self.simulation.boundary[0], 0, self.simulation.boundary[1]], cmap='Blues', alpha=0.5)

        if len(self.simulation.zombies) > 0:
            zombie_positions = np.array([z.position for z in self.simulation.zombies])
            H, _, _ = np.histogram2d(zombie_positions[:, 0], zombie_positions[:, 1], bins=[xedges, yedges])
            self.ax2.imshow(H.T, origin='lower', extent=[0, self.simulation.boundary[0], 0, self.simulation.boundary[1]], cmap='Reds', alpha=0.5)

        self.ax2.set_title('Density Heatmap of Survivors and Zombies', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('X Position')
        self.ax2.set_ylabel('Y Position')

    def update_health_plot(self):
        self.ax3.clear()
        for idx, survivor in enumerate(self.simulation.survivors):
            if survivor.health_history:
                self.ax3.plot(survivor.health_history, label=f"Survivor {idx+1}", linestyle='-', linewidth=2)

        self.ax3.set_xlabel('Step', fontsize=14)
        self.ax3.set_ylabel('Health', fontsize=14)
        self.ax3.set_title('Survivor Health Over Time', fontsize=14, fontweight='bold')
        self.ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax3.legend()

    def update_speed_plot(self):
        self.ax4.clear()

        # Calculate speeds of survivors and zombies
        survivor_speeds = [round(np.linalg.norm(survivor.velocity), 2) for survivor in self.simulation.survivors]
        zombie_speeds = [round(np.linalg.norm(zombie.velocity), 2) for zombie in self.simulation.zombies]

        # Plot histograms for the speeds of survivors and zombies
        self.ax4.hist(survivor_speeds, bins=10, alpha=0.7, color='blue', label='Survivors')
        self.ax4.hist(zombie_speeds, bins=10, alpha=0.7, color='red', label='Zombies')

        self.ax4.set_title('Speed Distribution of Survivors and Zombies', fontsize=14, fontweight='bold')
        self.ax4.set_xlabel('Speed')
        self.ax4.set_ylabel('Count')
        self.ax4.legend()

    def update_plot(self):
        self.update_position_plot()
        self.update_health_plot()
        self.update_speed_plot()
        self.update_distribution_plot()
        self.fig.canvas.draw()
        plt.tight_layout()

    def update_performance_metrics(self):
        survival_time = self.simulation.steps * self.simulation.dt

        # Calculate the average distance between all pairs of survivors and zombies
        total_distance = sum(distance(survivor, zombie) for survivor in self.simulation.survivors for zombie in self.simulation.zombies)
        avg_distance = total_distance / (len(self.simulation.survivors) * len(self.simulation.zombies))

        # Update labels
        self.survival_time_label.config(text=f"Survival Time: {survival_time:.2f}s")
        self.distance_label.config(text=f"Avg. Distance Between: {avg_distance:.2f}")

        health_text = " | ".join(f"S{idx+1}: {survivor.health}" for idx, survivor in enumerate(self.simulation.survivors))
        self.health_label.config(text=f"Survivor Health: {health_text}")

        self.count_label.config(text=f"Survivors: {len(self.simulation.survivors)} | Zombies: {len(self.simulation.zombies)}")

        if self.simulation.survivors and len(self.simulation.survivors[0].health_history) > 1:
            health_depletion_rate = sum(survivor.health_history[0] - survivor.health_history[-1] for survivor in self.simulation.survivors) / survival_time
            self.health_depletion_label.config(text=f"Health Depletion Rate: {health_depletion_rate:.2f}")

    def setup_refresh(self):
        self.root.after(100, self.refresh)

    def refresh(self):
        if not self.simulation.paused or self.simulation.steps == 0:
            self.update_canvas()
            self.update_plot()
            self.update_performance_metrics()

            with self.simulation.update_condition:
                self.simulation.update_condition.notify()

        self.root.after(100, self.refresh)

    def start(self):
        tk.mainloop()

# Main Function
def main():
    simulation = Simulation()
    gui = GUI(simulation)
    gui.start()

if __name__ == "__main__":
    main()