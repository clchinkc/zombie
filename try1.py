import random
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
    return np.linalg.norm(entity1.position - entity2.position)

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
        if not self.paused:
            self.update_simulation()

    def pause(self):
        self.paused = True

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
        self.throttle_job = None
        self.throttle_interval = 500

    def setup_control_panel(self):
        self.control_panel = tk.Frame(self.root)
        self.control_panel.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = tk.Button(self.control_panel, text="Start", command=self.start_simulation)
        self.pause_button = tk.Button(self.control_panel, text="Pause", command=self.pause_simulation)
        self.reset_button = tk.Button(self.control_panel, text="Reset", command=self.reset_simulation)
        self.step_button = tk.Button(self.control_panel, text="Step", command=self.step_simulation)

        self.start_button.pack(side=tk.LEFT)
        self.pause_button.pack(side=tk.LEFT)
        self.reset_button.pack(side=tk.LEFT)
        self.step_button.pack(side=tk.LEFT)

        # Sliders for interactive controls
        self.speed_slider = ttk.Scale(self.control_panel, from_=0, to=10, orient="horizontal", command=self.update_speed)
        self.speed_slider.pack(side=tk.LEFT)
        self.speed_label = tk.Label(self.control_panel, text="Speed: 0")
        self.speed_label.pack(side=tk.LEFT)

    def update_speed_debounced(self):
        speed = self.speed_slider.get()
        self.speed_label.config(text=f"Speed: {speed:.2f}")
        self.simulation.survivor_speed = speed

    def update_speed(self, event):
        if self.debounce_job is not None:
            self.root.after_cancel(self.debounce_job)
        self.debounce_job = self.root.after(100, self.update_speed_debounced)

    def reset_throttle(self):
        self.throttle_job = None

    def throttled_start_simulation(self):
        if not self.throttle_job:
            self.start_simulation()
            self.throttle_job = self.root.after(self.throttle_interval, self.reset_throttle)

    def start_simulation(self):
        self.simulation.resume()
        self.start_button.config(state='disabled')
        self.pause_button.config(state='normal')

    def throttled_pause_simulation(self):
        if not self.throttle_job:
            self.pause_simulation()
            self.throttle_job = self.root.after(self.throttle_interval, self.reset_throttle)

    def pause_simulation(self):
        self.simulation.pause()
        self.pause_button.config(state='disabled')
        self.start_button.config(state='normal')

    def throttled_reset_simulation(self):
        if not self.throttle_job:
            self.reset_simulation()
            self.throttle_job = self.root.after(self.throttle_interval, self.reset_throttle)

    def reset_simulation(self):
        self.simulation.reset()
        self.canvas.delete("all")
        self.update_canvas()
        self.update_plot()
        self.update_performance_metrics()

    def throttled_step_simulation(self):
        if not self.throttle_job:
            self.step_simulation()
            self.throttle_job = self.root.after(self.throttle_interval, self.reset_throttle)

    def step_simulation(self):
        self.simulation.step_forward()
        self.update_canvas()
        self.update_plot()

    def setup_canvas(self):
        # Adjust canvas size and positioning to maximize space usage
        self.canvas = tk.Canvas(self.root, width=self.simulation.boundary[0]*10, height=self.simulation.boundary[1]*10)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_plot(self):
        # Adjust plot size and positioning
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 8), dpi=100)
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

        self.survival_time_label.pack()
        self.distance_label.pack()
        self.health_label.pack()
        self.count_label.pack()
        self.health_depletion_label.pack()
        self.performance_frame.pack(side=tk.TOP, fill=tk.X)

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

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()

        # Plot for the positions
        for i, survivor in enumerate(self.simulation.survivors):
            # Plot the movement history of each survivor
            self.ax1.plot(*zip(*survivor.position_history), color="blue", label="Survivors" if i == 0 else "")

        for i, zombie in enumerate(self.simulation.zombies):
            # Plot the movement history of each zombie
            self.ax1.plot(*zip(*zombie.position_history), color="red", label="Zombies" if i == 0 else "")

        if self.simulation.health_drop_positions:
            self.ax1.scatter(*zip(*self.simulation.health_drop_positions), color="purple", marker='x', label="Health Drop")

        self.ax1.set_xlabel('X Position', fontsize=14)
        self.ax1.set_ylabel('Y Position', fontsize=14)
        self.ax1.set_title('Survivor vs Zombie Position', fontsize=16, fontweight='bold')
        self.ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax1.legend(loc='upper right')

        # Plot for the health of each survivor
        for idx, survivor in enumerate(self.simulation.survivors):
            if survivor.health_history:  # Check if health history data is available
                self.ax2.plot(survivor.health_history, label=f"Survivor {idx+1}", linestyle='-', linewidth=2)

        self.ax2.set_xlabel('Step', fontsize=14)
        self.ax2.set_ylabel('Health', fontsize=14)
        self.ax2.set_title('Survivor Health Over Time', fontsize=16, fontweight='bold')
        self.ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        self.fig.canvas.draw()

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
        self.simulation.run()
        self.update_canvas()
        self.update_plot()
        self.update_performance_metrics()
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