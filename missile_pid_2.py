import sys
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class PIDController:
    def __init__(self, kp, ki, kd, control_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.control_limit = control_limit
        self.integral = 0
        self.prev_error = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Apply control limit if defined
        if self.control_limit is not None:
            control_signal = np.clip(control_signal, -self.control_limit, self.control_limit)
        return control_signal

class AdaptivePIDController:
    def __init__(self, kp, ki, kd, kp_adapt, ki_adapt, kd_adapt, control_limit=None):
        self.base_kp, self.base_ki, self.base_kd = kp, ki, kd
        self.kp_adapt, self.ki_adapt, self.kd_adapt = kp_adapt, ki_adapt, kd_adapt
        self.control_limit = control_limit
        self.integral = self.prev_error = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # Adaptive tuning of PID parameters based on error magnitude
        kp = self.base_kp + self.kp_adapt * np.abs(error)
        ki = self.base_ki + self.ki_adapt * np.abs(error)
        kd = self.base_kd + self.kd_adapt * np.abs(error)

        control_signal = kp * error + ki * self.integral + kd * derivative
        self.prev_error = error

        if self.control_limit is not None:
            control_signal = np.clip(control_signal, -self.control_limit, self.control_limit)
        return control_signal

def missile_model(x, y, altitude, control_signals, dt, disturbance):
    # Incorporating robust control by handling disturbances
    x += control_signals['x'] * dt + disturbance
    y += control_signals['y'] * dt + disturbance
    altitude += control_signals['z'] * dt - 9.81 * dt + disturbance
    return x, y, altitude

def random_disturbance(strength=0.1):
    return np.random.normal(0, strength)

# Simulation Function
# Global variable to track if the window is open
window_open = True

def on_window_close():
    global window_open
    window_open = False
    window.destroy()

def run_simulation():
    global running, current_x, current_y, current_altitude, setpoint_x, setpoint_y, setpoint_altitude, window_open

    if running:
        return  # Prevent multiple instances of the simulation

    running = True
    run_button.config(state=tk.DISABLED)  # Disable the run button to prevent re-clicks

    # Simulation parameters
    setpoint_altitude = 1000  # Desired altitude (meters)
    setpoint_x = 100  # Desired x position (meters)
    setpoint_y = 100  # Desired y position (meters)

    current_x, current_y, current_altitude = 0, 0, 500  # Initial positions
    x_data.clear()
    y_data.clear()
    z_data.clear()

    pid_x = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
    pid_y = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
    pid_z = PIDController(kp=0.6, ki=0.1, kd=0.2, control_limit=100)  # Altitude control
    # pid_x = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
    # pid_y = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
    # pid_z = AdaptivePIDController(kp=0.6, ki=0.1, kd=0.2, kp_adapt=0.01, ki_adapt=0.002, kd_adapt=0.01, control_limit=100)  # Altitude control

    time_step = 0.1
    n_steps = 500

    # Define additional data structures for logging
    velocities_x, velocities_y, velocities_z = [], [], []
    accelerations = []
    control_inputs_x, control_inputs_y, control_inputs_z = [], [], []
    environmental_conditions = []

    # Data for plotting
    time_points = np.linspace(0, n_steps * time_step, n_steps)
    x_positions, y_positions, altitudes = [], [], []
    x_errors, y_errors, z_errors = [], [], []

    # Initial states for velocity calculation
    prev_x, prev_y, prev_altitude = current_x, current_y, current_altitude

    # Simulation loop
    for i in time_points:
        disturbance = random_disturbance()
        environmental_conditions.append(disturbance)

        control_signal_x = pid_x.update(setpoint_x, current_x, time_step)
        control_signal_y = pid_y.update(setpoint_y, current_y, time_step)
        control_signal_z = pid_z.update(setpoint_altitude, current_altitude, time_step)

        control_inputs_x.append(control_signal_x)
        control_inputs_y.append(control_signal_y)
        control_inputs_z.append(control_signal_z)

        new_x, new_y, new_altitude = missile_model(
            current_x, current_y, current_altitude,
            {'x': control_signal_x, 'y': control_signal_y, 'z': control_signal_z},
            time_step, disturbance
        )

        # Calculate velocity and acceleration
        velocity_x = (new_x - prev_x) / time_step
        velocity_y = (new_y - prev_y) / time_step
        velocity_z = (new_altitude - prev_altitude) / time_step

        velocities_x.append(velocity_x)
        velocities_y.append(velocity_y)
        velocities_z.append(velocity_z)

        acceleration = (velocity_z - (prev_altitude - current_altitude) / time_step) / time_step
        accelerations.append(acceleration)

        # Update positions and errors
        current_x, current_y, current_altitude = new_x, new_y, new_altitude
        x_positions.append(current_x)
        y_positions.append(current_y)
        altitudes.append(current_altitude)

        x_errors.append(setpoint_x - current_x)
        y_errors.append(setpoint_y - current_y)
        z_errors.append(setpoint_altitude - current_altitude)

        # Prepare data for GUI update
        data = {
            'x_position': current_x,
            'y_position': current_y,
            'altitude': current_altitude,
            'velocity_x': velocities_x[-1] if velocities_x else 0,
            'velocity_y': velocities_y[-1] if velocities_y else 0,
            'velocity_z': velocities_z[-1] if velocities_z else 0,
            'control_x': control_inputs_x[-1] if control_inputs_x else 0,
            'control_y': control_inputs_y[-1] if control_inputs_y else 0,
            'control_z': control_inputs_z[-1] if control_inputs_z else 0,
            'time': i * time_step
        }

        # Check if the window is still open before updating GUI
        if not window_open:
            break  # Break out of the loop if the window is closed

        # Update GUI and Tkinter event processing
        window.after(0, update_gui, data)
        window.update_idletasks()
        window.update()

    running = False
    if window_open:
        run_button.config(state=tk.NORMAL)  # Re-enable the run button only if the window still exists


def update_gui(data):
    # Update labels with new data
    position_label.config(text=f"Position: ({data['x_position']:.2f}, {data['y_position']:.2f}, {data['altitude']:.2f}) m")
    velocity_label.config(text=f"Velocity: ({data['velocity_x']:.2f}, {data['velocity_y']:.2f}, {data['velocity_z']:.2f}) m/s")
    control_label.config(text=f"Control Inputs: ({data['control_x']:.2f}, {data['control_y']:.2f}, {data['control_z']:.2f})")
    time_label.config(text=f"Simulation Time: {data['time']:.2f} s")

    # Update the plot
    x_data.append(data['x_position'])
    y_data.append(data['y_position'])
    z_data.append(data['altitude'])
    ax.clear()
    ax.plot(x_data, y_data, z_data, label='Missile Trajectory')
    ax.scatter(setpoint_x, setpoint_y, setpoint_altitude, color='r', marker='o', label='Target')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Missile Trajectory')
    ax.legend()
    canvas.draw()
    
    # Update the plot
    x_data.append(data['x_position'])
    y_data.append(data['y_position'])
    z_data.append(data['altitude'])
    ax.clear()
    ax.plot(x_data, y_data, z_data, label='Missile Trajectory')
    ax.scatter(setpoint_x, setpoint_y, setpoint_altitude, color='r', marker='o', label='Target')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Missile Trajectory')
    ax.legend()
    canvas.draw()

# Tkinter setup
window = tk.Tk()
window.title('Missile Trajectory Simulation')
window.protocol("WM_DELETE_WINDOW", on_window_close)

# Labels for data display
position_label = tk.Label(window, text="Position: ")
position_label.pack()

velocity_label = tk.Label(window, text="Velocity: ")
velocity_label.pack()

control_label = tk.Label(window, text="Control Inputs: ")
control_label.pack()

time_label = tk.Label(window, text="Simulation Time: ")
time_label.pack()

# Matplotlib setup for plotting
fig = Figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=window)
widget = canvas.get_tk_widget()
widget.pack()

# Data for plotting
x_data, y_data, z_data = [], [], []

# Simulation running state
running = False

# Start Simulation Button
run_button = tk.Button(window, text="Start Simulation", command=run_simulation)
run_button.pack()

window.mainloop()