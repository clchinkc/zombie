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
        self.propulsion_end_position = None  # This will store the position where propulsion ends

    def update_state(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def plot_trajectory(self, target):
        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Missile Trajectory')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title('Missile Trajectory')
        plt.grid(True)

        if self.propulsion_end_position is not None:
            plt.scatter(*self.propulsion_end_position, color='orange', s=50, label='Propulsion End', zorder=5)

        target.plot()

        plt.legend()
        plt.show()


class BallisticMissile(Missile):
    def __init__(self, mass, fuel_mass, cross_section_area, drag_coefficient, launch_angle, max_thrust, burn_time, fuel_consumption_rate):
        super().__init__(mass, cross_section_area, drag_coefficient)
        self.launch_angle = launch_angle
        self.max_thrust = max_thrust  # The maximum thrust when the missile has full fuel
        self.burn_time = burn_time
        self.fuel_mass = fuel_mass  # The initial fuel mass
        self.fuel_consumption_rate = fuel_consumption_rate  # Rate at which fuel is consumed (kg/s)
    
    def apply_gravity(self):
        return np.array([0, -self.mass * G])

    def apply_air_resistance(self):
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:  # Check to avoid division by zero
            drag_direction = self.velocity / velocity_magnitude
        else:
            drag_direction = np.array([0., 0.])
        return -0.5 * RHO * velocity_magnitude ** 2 * self.Cd * self.A * drag_direction

    def apply_propulsion(self):
        angle_rad = np.radians(self.launch_angle)
        if self.time_elapsed <= self.burn_time and self.fuel_mass > 0:
            current_thrust = self.max_thrust * (self.fuel_mass / (self.mass - self.fuel_mass))
            # Decrease fuel mass
            self.fuel_mass -= self.fuel_consumption_rate * DT
            # Ensure fuel mass doesn't become negative
            self.fuel_mass = max(self.fuel_mass, 0)
            # Calculate the propulsion force
            return np.array([current_thrust * np.cos(angle_rad), current_thrust * np.sin(angle_rad)])
        else:
            return np.array([0, 0])

    def update_state(self):
        # Before updating state, adjust mass for fuel consumption
        self.mass -= self.fuel_consumption_rate * DT
        self.mass = max(self.mass, 0)  # Ensure mass doesn't become negative

        gravity_force = self.apply_gravity()
        drag_force = self.apply_air_resistance()
        thrust_force = self.apply_propulsion()
        total_force = gravity_force + drag_force + thrust_force
        acceleration = total_force / self.mass

        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.trajectory.append(self.position.copy())
        self.time_elapsed += DT

        if self.time_elapsed > self.burn_time and self.propulsion_end_position is None:
            # Store the position where propulsion ends
            self.propulsion_end_position = self.position.copy()


class CruiseMissile(Missile):
    def __init__(self, mass, cross_section_area, drag_coefficient, engine_thrust, fuel_mass, fuel_consumption_rate, target, cruise_altitude):
        super().__init__(mass, cross_section_area, drag_coefficient)
        self.engine_thrust = engine_thrust
        self.fuel_mass = fuel_mass
        self.fuel_consumption_rate = fuel_consumption_rate
        self.is_propulsion_active = True
        self.target = target
        self.cruise_altitude = cruise_altitude  # The altitude the missile should maintain above the ground

    def apply_gravity(self):
        return np.array([0, -self.mass * G])

    def apply_air_resistance(self):
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            drag_direction = self.velocity / velocity_magnitude
        else:
            drag_direction = np.array([0., 0.])
        return -0.5 * RHO * velocity_magnitude ** 2 * self.Cd * self.A * drag_direction

    def apply_propulsion(self):
        if self.is_propulsion_active and self.fuel_mass > 0:
            # Propulsion only adjusts horizontally in this simplified model
            return np.array([self.engine_thrust, 0])
        else:
            self.is_propulsion_active = False
            return np.array([0, 0])

    def update_state(self):
        if self.is_propulsion_active:
            self.mass -= self.fuel_consumption_rate * DT
            self.mass = max(self.mass, 0)

        # Apply forces
        gravity_force = self.apply_gravity()
        drag_force = self.apply_air_resistance()
        thrust_force = self.apply_propulsion()

        # Apply guidance corrections
        if self.is_propulsion_active:
            correction = self.guidance_correction()
            thrust_force += correction

        total_force = gravity_force + drag_force + thrust_force
        acceleration = total_force / self.mass

        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.trajectory.append(self.position.copy())
        self.time_elapsed += DT

        # Terrain-following logic to maintain cruise altitude
        self.position[1] = self.cruise_altitude

    def guidance_correction(self):
        # A simple proportional navigation logic to adjust the missile's heading towards the target
        target_vector = self.target.position - self.position
        heading = np.arctan2(target_vector[1], target_vector[0])

        # Calculate the current heading from velocity
        current_heading = np.arctan2(self.velocity[1], self.velocity[0])

        # Calculate the required heading change
        heading_change = heading - current_heading

        # Simplified control: Adjust the thrust vector based on the heading change
        # This is a simple and naive implementation and does not represent a real-world control system
        correction_strength = 0.1  # A constant to limit the rate of change
        correction_vector = np.array([np.cos(current_heading + heading_change * correction_strength),
                                      np.sin(current_heading + heading_change * correction_strength)])

        # Return the correction force vector
        return correction_vector * self.engine_thrust


class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < 10

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')
        plt.legend()

def objective(params, target, cruise_altitude):
    thrust, = params
    # Add a reasonable assumption for the initial fuel mass and consumption rate
    fuel_mass = 300  # Initial fuel mass in kg
    fuel_consumption_rate = 5  # Fuel consumption rate in kg/s
    
    # Instantiate the missile with additional parameters
    missile = CruiseMissile(mass=1000, fuel_mass=fuel_mass, cross_section_area=0.3, drag_coefficient=0.5, 
                            engine_thrust=thrust, fuel_consumption_rate=fuel_consumption_rate,
                            target=target, cruise_altitude=cruise_altitude)
    
    while missile.fuel_mass > 0:
        missile.update_state()
        if target.is_hit_by(missile):
            return 0  # Target hit

    distance_to_target = np.linalg.norm(missile.position - target.position)
    return distance_to_target

def find_optimal_conditions(target, cruise_altitude):
    result = minimize(objective, [200000], args=(target, cruise_altitude), bounds=[(0, 500000)])
    return result.x

def main():
    target_position = [10000, 0]  # Assuming 0 is the ground level
    cruise_altitude = 1000  # Altitude at which the missile should cruise
    target = Target(target_position)

    optimal_thrust = find_optimal_conditions(target, cruise_altitude)[0]
    print(f"Optimal Thrust: {optimal_thrust:.2f} N")

    missile = CruiseMissile(mass=1000, fuel_mass=300, cross_section_area=0.3, drag_coefficient=0.5, 
                            engine_thrust=optimal_thrust, fuel_consumption_rate=5,
                            target=target, cruise_altitude=cruise_altitude)
    
    while missile.fuel_mass > 0:
        missile.update_state()

    missile.plot_trajectory(target)

if __name__ == "__main__":
    main()
