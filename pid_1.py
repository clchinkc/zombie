import matplotlib.pyplot as plt
import numpy as np
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

class CascadePIDController:
    def __init__(self, primary_pid, secondary_pid):
        self.primary_pid = primary_pid
        self.secondary_pid = secondary_pid

    def update(self, primary_setpoint, primary_measured, secondary_measured, dt):
        secondary_setpoint = self.primary_pid.update(primary_setpoint, primary_measured, dt)
        control_signal = self.secondary_pid.update(secondary_setpoint, secondary_measured, dt)
        return control_signal

def missile_model(x, y, altitude, control_signals, dt, disturbance):
    # Incorporating robust control by handling disturbances
    x += control_signals['x'] * dt + disturbance
    y += control_signals['y'] * dt + disturbance
    altitude += control_signals['z'] * dt - 9.81 * dt + disturbance
    return x, y, altitude

def random_disturbance(strength=0.1):
    return np.random.normal(0, strength)

# Simulation parameters
setpoint_altitude = 1000  # Desired altitude (meters)
setpoint_x = 100  # Desired x position (meters)
setpoint_y = 100  # Desired y position (meters)

current_x, current_y, current_altitude = 0, 0, 500  # Initial positions
current_thrust = 0  # Initial thrust, an intermediate control variable

pid_x = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_y = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_z = PIDController(kp=0.6, ki=0.1, kd=0.2, control_limit=100)  # Altitude control
# pid_x = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
# pid_y = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
# pid_z = AdaptivePIDController(kp=0.6, ki=0.1, kd=0.2, kp_adapt=0.01, ki_adapt=0.002, kd_adapt=0.01, control_limit=100)  # Altitude control

pid_thrust = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)  # Secondary PID for thrust
cascade_pid = CascadePIDController(pid_z, pid_thrust)

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
prev_x, prev_y, prev_altitude = 0, 0, 500

# Simulation loop
for _ in time_points:
    disturbance = random_disturbance()
    environmental_conditions.append(disturbance)

    control_signal_x = pid_x.update(setpoint_x, current_x, time_step)
    control_signal_y = pid_y.update(setpoint_y, current_y, time_step)
    # control_signal_z = pid_z.update(setpoint_altitude, current_altitude, time_step)
    control_signal_thrust = cascade_pid.update(setpoint_altitude, current_altitude, current_thrust, time_step)

    control_inputs_x.append(control_signal_x)
    control_inputs_y.append(control_signal_y)
    control_inputs_z.append(control_signal_thrust)

    # Update the missile model using the control signals
    new_x, new_y, new_altitude = missile_model(
        current_x, current_y, current_altitude,
        {'x': control_signal_x, 'y': control_signal_y, 'z': control_signal_thrust},
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

    # Update previous state
    prev_x, prev_y, prev_altitude = new_x, new_y, new_altitude

# 3D Plotting and additional plots
fig = plt.figure(figsize=(14, 10))

# Trajectory Plot
ax1 = fig.add_subplot(321, projection='3d')
ax1.plot(x_positions, y_positions, altitudes, label='Missile Trajectory')
ax1.scatter(setpoint_x, setpoint_y, setpoint_altitude, color='r', marker='o', label='Target')
ax1.set_xlabel('X Position (meters)')
ax1.set_ylabel('Y Position (meters)')
ax1.set_zlabel('Altitude (meters)')
ax1.set_title('3D Trajectory of Missile')
ax1.legend()

# Error Plot
ax2 = fig.add_subplot(322)
ax2.plot(time_points, x_errors, label='X Error')
ax2.plot(time_points, y_errors, label='Y Error')
ax2.plot(time_points, z_errors, label='Altitude Error')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Error (meters)')
ax2.set_title('Control Errors Over Time')
ax2.legend()

# Velocity Plot
ax3 = fig.add_subplot(323)
ax3.plot(time_points, velocities_x, label='X Velocity')
ax3.plot(time_points, velocities_y, label='Y Velocity')
ax3.plot(time_points, velocities_z, label='Z Velocity')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity Over Time')
ax3.legend()

# Acceleration Plot
ax4 = fig.add_subplot(324)
ax4.plot(time_points, accelerations, label='Acceleration')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Acceleration (m/s²)')
ax4.set_title('Acceleration Over Time')
ax4.legend()

# Control Input Plot
ax5 = fig.add_subplot(325)
ax5.plot(time_points, control_inputs_x, label='Control Input X')
ax5.plot(time_points, control_inputs_y, label='Control Input Y')
ax5.plot(time_points, control_inputs_z, label='Control Input Z')
ax5.set_xlabel('Time (seconds)')
ax5.set_ylabel('Control Input')
ax5.set_title('Control Inputs Over Time')
ax5.legend()

# Environmental Conditions Plot
ax6 = fig.add_subplot(326)
ax6.plot(time_points, environmental_conditions, label='Environmental Disturbance')
ax6.set_xlabel('Time (seconds)')
ax6.set_ylabel('Disturbance')
ax6.set_title('Environmental Conditions Over Time')
ax6.legend()

plt.tight_layout()
plt.show()