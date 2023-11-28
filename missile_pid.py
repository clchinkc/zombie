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
        self.prev_error = None

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        if self.prev_error is None:
            derivative = 0
        else:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative
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

def missile_model(x, y, z, control_signals, dt, disturbance):
    # Incorporating robust control by handling disturbances
    x += control_signals['x'] * dt + disturbance
    y += control_signals['y'] * dt + disturbance
    z += control_signals['z'] * dt - 9.81 * dt + disturbance
    return x, y, z

def random_disturbance(strength=0.1):
    return np.random.normal(0, strength)

# Global variables for the simulation
running = False
window_open = True

def on_window_close():
    global window_open
    window_open = False
    window.destroy()

def run_simulation():
    global running, current_x, current_y, current_z, setpoint_x, setpoint_y, setpoint_z
    global x_positions, y_positions, z_positions, time_points, velocities_x, velocities_y, velocities_z
    global accelerations_x, accelerations_y, accelerations_z, control_inputs_x, control_inputs_y, control_inputs_z
    global x_errors, y_errors, z_errors, environmental_conditions

    if running:
        return

    running = True
    run_button.config(state=tk.DISABLED)

    # Simulation parameters
    setpoint_z = 1000  # Desired altitude (meters)
    setpoint_x = 100  # Desired x position (meters)
    setpoint_y = 100  # Desired y position (meters)

    current_x, current_y, current_z = 0, 0, 500  # Initial positions
    x_positions.clear()
    y_positions.clear()
    z_positions.clear()

    pid_x = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
    pid_y = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
    pid_z = PIDController(kp=0.6, ki=0.1, kd=0.1, control_limit=100)
    # pid_x = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
    # pid_y = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
    # pid_z = AdaptivePIDController(kp=0.6, ki=0.1, kd=0.2, kp_adapt=0.01, ki_adapt=0.002, kd_adapt=0.01, control_limit=100)  # Altitude control

    time_step = 0.1
    n_steps = 500
    time_points = np.linspace(0, n_steps * time_step, n_steps)

    velocities_x, velocities_y, velocities_z = [], [], []
    accelerations_x, accelerations_y, accelerations_z = [], [], []
    control_inputs_x, control_inputs_y, control_inputs_z = [], [], []
    x_errors, y_errors, z_errors = [], [], []
    environmental_conditions = []

    prev_x, prev_y, prev_z = current_x, current_y, current_z

    for i in time_points:
        disturbance = random_disturbance()
        environmental_conditions.append(disturbance)

        control_signal_x = pid_x.update(setpoint_x, current_x, time_step)
        control_signal_y = pid_y.update(setpoint_y, current_y, time_step)
        control_signal_z = pid_z.update(setpoint_z, current_z, time_step)

        control_inputs_x.append(control_signal_x)
        control_inputs_y.append(control_signal_y)
        control_inputs_z.append(control_signal_z)

        current_x, current_y, current_z = missile_model(
            current_x, current_y, current_z,
            {'x': control_signal_x, 'y': control_signal_y, 'z': control_signal_z},
            time_step, disturbance
        )
        
        x_positions.append(current_x)
        y_positions.append(current_y)
        z_positions.append(current_z)

        # Calculate velocity and acceleration
        velocity_x = (current_x - prev_x) / time_step
        velocity_y = (current_y - prev_y) / time_step
        velocity_z = (current_z - prev_z) / time_step

        velocities_x.append(velocity_x)
        velocities_y.append(velocity_y)
        velocities_z.append(velocity_z)

        acceleration_x = (velocity_x - (prev_x - current_x) / time_step) / time_step
        acceleration_y = (velocity_y - (prev_y - current_y) / time_step) / time_step
        acceleration_z = (velocity_z - (prev_z - current_z) / time_step) / time_step

        accelerations_x.append(acceleration_x)
        accelerations_y.append(acceleration_y)
        accelerations_z.append(acceleration_z)
        
        x_error = setpoint_x - current_x
        y_error = setpoint_y - current_y
        z_error = setpoint_z - current_z
        x_errors.append(x_error)
        y_errors.append(y_error)
        z_errors.append(z_error)

        prev_x, prev_y, prev_z = current_x, current_y, current_z

        data = {
            'x_position': current_x,
            'y_position': current_y,
            'z_position': current_z,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'acceleration_x': acceleration_x,
            'acceleration_y': acceleration_y,
            'acceleration_z': acceleration_z,
            'control_x': control_signal_x,
            'control_y': control_signal_y,
            'control_z': control_signal_z,
            'x_error': x_error,
            'y_error': y_error,
            'z_error': z_error,
            'time': i
        }

        if not window_open:
            break

        window.after(0, update_gui, data)
        window.update_idletasks()
        window.update()

    running = False
    if window_open:
        run_button.config(state=tk.NORMAL)  # Re-enable the run button only if the window still exists


def update_gui(data):
    position_label.config(text=f"Position: ({data['x_position']:.2f}, {data['y_position']:.2f}, {data['z_position']:.2f}) m")
    velocity_label.config(text=f"Velocity: ({data['velocity_x']:.2f}, {data['velocity_y']:.2f}, {data['velocity_z']:.2f}) m/s")
    acceleration_label.config(text=f"Acceleration: ({data['acceleration_x']:.2f}, {data['acceleration_y']:.2f}, {data['acceleration_z']:.2f}) m/s²")
    control_label.config(text=f"Control Inputs: ({data['control_x']:.2f}, {data['control_y']:.2f}, {data['control_z']:.2f})")
    error_label.config(text=f"Control Error: ({data['x_error']:.2f}, {data['y_error']:.2f}, {data['z_error']:.2f})")
    time_label.config(text=f"Simulation Time: {data['time']:.2f} s")

    ax_trajectory.clear()
    ax_trajectory.plot(x_positions, y_positions, z_positions, label='Missile Trajectory')
    ax_trajectory.scatter(setpoint_x, setpoint_y, setpoint_z, color='r', marker='o', label='Target')
    ax_trajectory.set_xlabel('X Position (m)')
    ax_trajectory.set_ylabel('Y Position (m)')
    ax_trajectory.set_zlabel('Altitude (m)')
    ax_trajectory.set_title('Missile Trajectory')
    ax_trajectory.legend()

    ax_position.clear()
    ax_position.plot(time_points[:len(x_positions)], x_positions, label='X Position')
    ax_position.plot(time_points[:len(y_positions)], y_positions, label='Y Position')
    ax_position.plot(time_points[:len(z_positions)], z_positions, label='Z Position')
    ax_position.set_xlabel('Time (s)')
    ax_position.set_ylabel('Position (m)')
    ax_position.set_title('Position Over Time')
    ax_position.legend()

    ax_velocity.clear()
    ax_velocity.plot(time_points[:len(velocities_x)], velocities_x, label='X Velocity')
    ax_velocity.plot(time_points[:len(velocities_y)], velocities_y, label='Y Velocity')
    ax_velocity.plot(time_points[:len(velocities_z)], velocities_z, label='Z Velocity')
    ax_velocity.set_xlabel('Time (s)')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_title('Velocity Over Time')
    ax_velocity.legend()

    ax_acceleration.clear()
    ax_acceleration.plot(time_points[:len(accelerations_x)], accelerations_x, label='X Acceleration')
    ax_acceleration.plot(time_points[:len(accelerations_y)], accelerations_y, label='Y Acceleration')
    ax_acceleration.plot(time_points[:len(accelerations_z)], accelerations_z, label='Z Acceleration')
    ax_acceleration.set_xlabel('Time (s)')
    ax_acceleration.set_ylabel('Acceleration (m/s²)')
    ax_acceleration.set_title('Acceleration Over Time')
    ax_acceleration.legend()

    ax_control.clear()
    ax_control.plot(time_points[:len(control_inputs_x)], control_inputs_x, label='X Control Signal')
    ax_control.plot(time_points[:len(control_inputs_y)], control_inputs_y, label='Y Control Signal')
    ax_control.plot(time_points[:len(control_inputs_z)], control_inputs_z, label='Z Control Signal')
    ax_control.set_xlabel('Time (s)')
    ax_control.set_ylabel('Control Signal')
    ax_control.set_title('Control Signal Over Time')
    ax_control.legend()
    
    ax_error.clear()
    ax_error.plot(time_points[:len(x_errors)], x_errors, label='X Error')
    ax_error.plot(time_points[:len(y_errors)], y_errors, label='Y Error')
    ax_error.plot(time_points[:len(z_errors)], z_errors, label='Z Error')
    ax_error.set_xlabel('Time (s)')
    ax_error.set_ylabel('Error')
    ax_error.set_title('Control Error Over Time')
    ax_error.legend()

    ax_environmental.clear()
    ax_environmental.plot(time_points[:len(environmental_conditions)], environmental_conditions, label='Disturbance')
    ax_environmental.set_xlabel('Time (s)')
    ax_environmental.set_ylabel('Disturbance')
    ax_environmental.set_title('Disturbance Over Time')
    ax_environmental.legend()

    canvas.draw()
    fig.tight_layout()

# Tkinter setup
window = tk.Tk()
window.title('Missile Trajectory Simulation')
window.protocol("WM_DELETE_WINDOW", on_window_close)

# Data for plotting
x_positions, y_positions, z_positions = [], [], []

# Frame for the control buttons and labels
control_frame = tk.Frame(window)
control_frame.pack(side=tk.RIGHT)

# Start Simulation Button
run_button = tk.Button(control_frame, text="Start Simulation", command=run_simulation)
run_button.pack(side=tk.TOP)

# Labels for data display
position_label = tk.Label(control_frame, text="Position: ")
position_label.pack(side=tk.TOP)
velocity_label = tk.Label(control_frame, text="Velocity: ")
velocity_label.pack(side=tk.TOP)
acceleration_label = tk.Label(control_frame, text="Acceleration: ")
acceleration_label.pack(side=tk.TOP)
control_label = tk.Label(control_frame, text="Control Inputs: ")
control_label.pack(side=tk.TOP)
error_label = tk.Label(control_frame, text="Control Error: ")
error_label.pack(side=tk.TOP)
time_label = tk.Label(control_frame, text="Simulation Time: ")
time_label.pack(side=tk.TOP)

# Matplotlib setup for plotting
fig = Figure(figsize=(12, 8))
ax_trajectory = fig.add_subplot(421, projection='3d')
ax_position = fig.add_subplot(422)
ax_velocity = fig.add_subplot(423)
ax_acceleration = fig.add_subplot(424)
ax_error = fig.add_subplot(425)
ax_control = fig.add_subplot(426)
ax_environmental = fig.add_subplot(427)
canvas = FigureCanvasTkAgg(fig, master=window)
widget = canvas.get_tk_widget()
widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

window.mainloop()


