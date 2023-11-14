import matplotlib.pyplot as plt
import numpy as np


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

def missile_model(x, y, altitude, control_signals, dt, disturbance):
    # Applying control signals and handling disturbances
    x += control_signals['x'] * dt + disturbance
    y += control_signals['y'] * dt + disturbance
    altitude += control_signals['z'] * dt - 9.81 * dt + disturbance
    return x, y, altitude

def random_disturbance(strength=0.1):
    return np.random.normal(0, strength)

# Simulation parameters
setpoint_altitude = 1000
setpoint_x = 100
setpoint_y = 100

current_x, current_y, current_altitude = 0, 0, 500
prev_x, prev_y, prev_altitude = 0, 0, 500

# Outer Loop PID Controllers (Position/Altitude Control)
pid_x_position = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_y_position = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_z_altitude = PIDController(kp=0.6, ki=0.1, kd=0.1, control_limit=100)

# Inner Loop PID Controllers (Velocity Control)
pid_x_velocity = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_y_velocity = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_z_velocity = PIDController(kp=0.6, ki=0.1, kd=0.1, control_limit=100)

time_step = 0.1
n_steps = 500
time_points = np.linspace(0, n_steps * time_step, n_steps)

# Data structures for plotting
x_positions, y_positions, altitudes = [], [], []
x_velocities, y_velocities, z_velocities = [], [], []

for _ in time_points:
    disturbance = random_disturbance()

    # Outer Loop Control
    velocity_setpoint_x = pid_x_position.update(setpoint_x, current_x, time_step)
    velocity_setpoint_y = pid_y_position.update(setpoint_y, current_y, time_step)
    velocity_setpoint_z = pid_z_altitude.update(setpoint_altitude, current_altitude, time_step)

    # Inner Loop Control
    control_signal_x = pid_x_velocity.update(velocity_setpoint_x, (current_x - prev_x) / time_step, time_step)
    control_signal_y = pid_y_velocity.update(velocity_setpoint_y, (current_y - prev_y) / time_step, time_step)
    control_signal_z = pid_z_velocity.update(velocity_setpoint_z, (current_altitude - prev_altitude) / time_step, time_step)

    current_x, current_y, current_altitude = missile_model(current_x, current_y, current_altitude,
                                                            {'x': control_signal_x, 'y': control_signal_y,
                                                             'z': control_signal_z}, time_step, disturbance)

    x_positions.append(current_x)
    y_positions.append(current_y)
    altitudes.append(current_altitude)
    x_velocities.append(control_signal_x)
    y_velocities.append(control_signal_y)
    z_velocities.append(control_signal_z)

    prev_x, prev_y, prev_altitude = current_x, current_y, current_altitude
    
# Plotting
fig = plt.figure(figsize=(14, 10))

# Trajectory Plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(x_positions, y_positions, altitudes, label='Missile Trajectory')
ax1.scatter(setpoint_x, setpoint_y, setpoint_altitude, color='r', marker='o', label='Target')
ax1.set_xlabel('X Position (meters)')
ax1.set_ylabel('Y Position (meters)')
ax1.set_zlabel('Altitude (meters)')
ax1.set_title('3D Trajectory of Missile')
ax1.legend()

# Velocity Plot
ax2 = fig.add_subplot(222)
ax2.plot(time_points, x_velocities, label='X Velocity')
ax2.plot(time_points, y_velocities, label='Y Velocity')
ax2.plot(time_points, z_velocities, label='Z Velocity')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Velocity Over Time')
ax2.legend()

plt.tight_layout()
plt.show()