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
        self.propulsion_end_position = None

    def apply_gravity(self):
        # Gravity applies to all missiles, but can be zero in some cases.
        return np.array([0., -self.mass * G])

    def apply_air_resistance(self):
        # Air resistance applies to all missiles, can be overridden if the calculation differs.
        velocity_magnitude = np.linalg.norm(self.velocity)
        drag_direction = self.velocity / velocity_magnitude if velocity_magnitude > 0 else np.array([0., 0.])
        return -0.5 * RHO * velocity_magnitude ** 2 * self.Cd * self.A * drag_direction

    def apply_propulsion(self):
        # This method will be specific to the subclass implementation.
        raise NotImplementedError("The apply_propulsion method should be implemented by subclasses")

    def update_forces(self):
        # Update forces. This will call the relevant methods that can be overridden by subclasses.
        forces = np.array([0., 0.])
        forces += self.apply_gravity()
        forces += self.apply_air_resistance()
        forces += self.apply_propulsion()
        return forces

    def update_state(self):
        total_force = self.update_forces()
        acceleration = total_force / self.mass
        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.trajectory.append(self.position.copy())
        self.time_elapsed += DT

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


class BallisticMissile(Missile):
    def __init__(self, mass, fuel_mass, cross_section_area, drag_coefficient, launch_angle, max_thrust, burn_time, fuel_consumption_rate):
        super().__init__(mass, cross_section_area, drag_coefficient)
        self.launch_angle = launch_angle
        self.max_thrust = max_thrust  # The maximum thrust when the missile has full fuel
        self.burn_time = burn_time
        self.fuel_mass = fuel_mass  # The initial fuel mass
        self.fuel_consumption_rate = fuel_consumption_rate  # Rate at which fuel is consumed (kg/s)

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
            return np.array([0., 0.])

    def update_state(self):
        self.mass -= self.fuel_consumption_rate * DT
        self.mass = max(self.mass, 0)
        super().update_state()

    def objective(self, target, fuel_efficiency_weight=1.0, time_penalty_weight=1.0):
        initial_fuel_mass = self.fuel_mass
        while self.velocity[1] >= 0 or self.position[1] > target.position[1]:
            self.update_state()
            if target.is_hit_by(self):
                return 0  # Target hit
        # If the missile has passed the target's altitude or fallen to the ground, compute the miss distance
        distance_to_target = np.linalg.norm(self.position - target.position)
        
        # Calculate the fuel efficiency penalty
        fuel_used = initial_fuel_mass - self.fuel_mass
        fuel_efficiency_penalty = fuel_efficiency_weight * fuel_used
        
        # Calculate time penalty
        time_penalty = time_penalty_weight * self.time_elapsed
        
        # Combine the distance, fuel efficiency penalty, and time penalty into a single objective
        return distance_to_target + fuel_efficiency_penalty + time_penalty


    @classmethod
    def find_optimal_conditions(cls, target, initial_guess, bounds):
        def callback(xk):
            print(f"Current iteration parameters: launch angle = {xk[0]}, thrust = {xk[1]}, burn time = {xk[2]}")

        def _objective(params):
            launch_angle, thrust, burn_time = params
            # Instantiate the missile with given parameters
            missile = cls(mass=1000, fuel_mass=300, cross_section_area=0.3, drag_coefficient=0.5, 
                          launch_angle=launch_angle, max_thrust=thrust, burn_time=burn_time, 
                          fuel_consumption_rate=5)
            return missile.objective(target)

        result = minimize(_objective, initial_guess, bounds=bounds, callback=callback)
        return result.x


class CruiseMissile(Missile):
    def __init__(self, mass, cross_section_area, drag_coefficient, engine_thrust, max_flight_time):
        super().__init__(mass, cross_section_area, drag_coefficient)
        self.engine_thrust = engine_thrust
        self.max_flight_time = max_flight_time
        self.boost_time = 10  # Boost phase duration in seconds
        self.terminal_phase_start = 1000  # Distance in meters from target to start terminal phase
        self.cruising_altitude = 500  # Cruising altitude in meters
        self.guidance_system = 'launch'  # Start with launch phase guidance

    def apply_propulsion(self):
        if self.guidance_system == 'launch':
            if self.time_elapsed <= self.boost_time:
                climb_angle_rad = np.radians(45)  # Climb angle set to 45 degrees
                vertical_thrust_component = self.engine_thrust * np.sin(climb_angle_rad)
                horizontal_thrust_component = self.engine_thrust * np.cos(climb_angle_rad)

                # Apply vertical thrust component to counter gravity and the rest to climb at a 45-degree angle
                return np.array([
                    horizontal_thrust_component,
                    vertical_thrust_component + self.mass * G  # Subtract gravity's force component
                ])
            else:
                # Transition to cruise phase after boost
                self.guidance_system = 'cruise'
                return np.array([self.engine_thrust, 0.])
        elif self.guidance_system == 'cruise':
            # Maintain the propulsion in the horizontal direction
            return np.array([self.engine_thrust, 0.])
        elif self.guidance_system == 'terminal':
            # No propulsion in terminal phase, missile is on final approach
            return np.array([0., 0.])

    def apply_gravity(self):
        if self.guidance_system == 'launch' and self.position[1] < self.cruising_altitude:
            return super().apply_gravity()
        else:
            # Gravity is countered by lift in cruise and terminal phases
            return np.array([0., 0.])

    def update_guidance(self, target):
        distance_to_target = np.linalg.norm(target.position - self.position)

        # Switch to terminal phase based on distance to target
        if distance_to_target <= self.terminal_phase_start and self.guidance_system != 'terminal':
            self.guidance_system = 'terminal'
            self.terminal_phase_init_velocity = np.linalg.norm(self.velocity)  # Capture initial velocity for terminal phase

        if self.guidance_system == 'launch':
            # Launch phase guidance logic
            if self.position[1] >= self.cruising_altitude:
                self.guidance_system = 'cruise'
        elif self.guidance_system == 'cruise':
            # Cruise phase guidance logic
            direction_to_target = (target.position - self.position) / distance_to_target
            # Adjust only the horizontal velocity
            self.velocity[0] = direction_to_target[0] * np.linalg.norm(self.velocity)
            # Maintain low altitude flight
            self.velocity[1] = 0
            self.position[1] = self.cruising_altitude
        elif self.guidance_system == 'terminal':
            # Calculate the horizontal and vertical distance to the target
            horizontal_distance = target.position[0] - self.position[0]
            vertical_distance = self.position[1] - target.position[1]

            # Start with a gentle descent and increase the steepness as we get closer to the target
            # We'll use a function of the horizontal distance to calculate a descent angle that
            # gets steeper as the missile approaches the target.
            # This factor will increase from 0 to 1 as the missile approaches the target
            steepness_factor = np.clip(1 - horizontal_distance / self.terminal_phase_start, 0, 1)
            # Use an arctangent function to get an angle that increases as we get closer
            descent_angle_rad = np.arctan(steepness_factor * (vertical_distance / horizontal_distance))

            # Adjust velocity towards the descent angle
            self.velocity[0] = np.cos(descent_angle_rad) * np.linalg.norm(self.velocity)
            self.velocity[1] = -np.sin(descent_angle_rad) * np.linalg.norm(self.velocity)

    def update_state(self, target):
        self.update_guidance(target)
        super().update_state()


class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < 10

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')
        plt.legend()

def main():
    target_position = [10000, 1000]
    target = Target(target_position)

    initial_guess = [45, 200000, 10]  # Initial guess for the optimization
    bounds = [(0, 90), (0, 500000), (0, 60)]  # Bounds for launch_angle, thrust, and burn_time
    optimal_launch_angle, optimal_thrust, optimal_burn_time = BallisticMissile.find_optimal_conditions(target, initial_guess, bounds)

    # Create a missile instance with the optimal conditions found
    missile = BallisticMissile(
        mass=1000, fuel_mass=300, cross_section_area=0.3, drag_coefficient=0.5,
        launch_angle=optimal_launch_angle, max_thrust=optimal_thrust, burn_time=optimal_burn_time,
        fuel_consumption_rate=5
    )
    
    while (missile.position[0] < target.position[0] or missile.position[1] >= 0 or missile.position[1] < target.position[1] or missile.position[0] >= target.position[0]) and not target.is_hit_by(missile):
        missile.update_state()

    missile.plot_trajectory(target)
    
    target_position = [10000, 100]  # Assume a flat terrain and target at sea level
    target = Target(target_position)

    # Create the target and cruise missile
    cruise_missile = CruiseMissile(mass=1000, cross_section_area=0.3, drag_coefficient=0.5, engine_thrust=5000, max_flight_time=9600)
    
    # Update and plot the cruise missile's trajectory
    while cruise_missile.time_elapsed < cruise_missile.max_flight_time:
        cruise_missile.update_state(target)
        if target.is_hit_by(cruise_missile):
            break

    cruise_missile.plot_trajectory(target)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
