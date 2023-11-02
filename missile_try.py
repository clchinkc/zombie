import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

G = 9.81  # Gravitational acceleration (m/s^2)
RHO = 1.225  # Air density at sea level (kg/m^3)
DT = 0.01  # Time step (s)

class Missile:
    def __init__(self, mass, cross_section_area, drag_coefficient):
        self.mass = mass
        self.A = cross_section_area
        self.Cd = drag_coefficient
        self.position = np.array([0., 0.])
        self.velocity = np.array([0., 0.])
        self.trajectory = [self.position.copy()]
        self.time_elapsed = 0.0

    def update_state(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def plot_trajectory(self):
        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title('Missile Trajectory')
        plt.grid(True)
        plt.show()


class BallisticMissile(Missile):
    def __init__(self, mass, cross_section_area, drag_coefficient, launch_angle, thrust, burn_time):
        super().__init__(mass, cross_section_area, drag_coefficient)
        self.launch_angle = launch_angle
        self.thrust = thrust
        self.burn_time = burn_time

    def apply_gravity(self):
        return np.array([0, -self.mass * G])

    def apply_air_resistance(self):
        velocity_magnitude = np.linalg.norm(self.velocity)
        return -0.5 * RHO * velocity_magnitude * self.velocity * self.Cd * self.A

    def apply_propulsion(self):
        angle_rad = np.radians(self.launch_angle)
        if self.time_elapsed <= self.burn_time:
            return np.array([self.thrust * np.cos(angle_rad), self.thrust * np.sin(angle_rad)])
        else:
            return np.array([0, 0])

    def update_state(self):
        gravity_force = self.apply_gravity()
        drag_force = self.apply_air_resistance()
        thrust_force = self.apply_propulsion()
        total_force = gravity_force + drag_force + thrust_force
        acceleration = total_force / self.mass

        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.trajectory.append(self.position.copy())
        self.time_elapsed += DT


class CruiseMissile(Missile):
    def __init__(self, mass, cross_section_area, drag_coefficient):
        super().__init__(mass, cross_section_area, drag_coefficient)
        # Additional attributes specific to cruise missile can be added here

    def update_state(self):
        # Implementation specific to cruise missile goes here
        pass  # Placeholder for now

class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < 10

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')
        plt.legend()

def objective(params, target):
    launch_angle, thrust, burn_time = params
    missile = BallisticMissile(mass=1000, cross_section_area=0.3, drag_coefficient=0.5, launch_angle=launch_angle, thrust=thrust, burn_time=burn_time)
    
    while missile.position[1] >= 0:
        missile.update_state()
        if target.is_hit_by(missile):
            return 0  # Target hit

    distance_to_target = np.linalg.norm(missile.position - target.position)
    return distance_to_target

def find_optimal_conditions(target):
    result = minimize(objective, [45, 200000, 10], args=(target), bounds=[(0, 90), (0, 500000), (0, 60)])
    return result.x

def main():
    target_position = [10000, 0]
    target = Target(target_position)

    optimal_launch_angle, optimal_thrust, optimal_burn_time = find_optimal_conditions(target)
    print(f"Optimal Launch Angle: {optimal_launch_angle:.2f} degrees")
    print(f"Optimal Thrust: {optimal_thrust:.2f} N")
    print(f"Optimal Burn Time: {optimal_burn_time:.2f} s")

    missile = BallisticMissile(mass=1000, cross_section_area=0.3, drag_coefficient=0.5, launch_angle=optimal_launch_angle, thrust=optimal_thrust, burn_time=optimal_burn_time)
    while missile.position[1] >= 0:
        missile.update_state()

    missile.plot_trajectory()
    target.plot()

if __name__ == "__main__":
    main()
