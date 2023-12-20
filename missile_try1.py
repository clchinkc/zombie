import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

GRAVITY = 9.81  # Gravitational acceleration (m/s^2)
AIR_DENSITY = 1.225  # Air density at sea level (kg/m^3)
TIME_STEP = 0.01  # Time step (s)
HIT_RADIUS = 10  # Radius within which the target is considered hit (m)

class Missile:
    def __init__(self, target, mass, cross_section_area, drag_coefficient):
        self.target = target
        self.mass = mass
        self.A = cross_section_area
        self.Cd = drag_coefficient
        self.position = np.array([0., 0.])
        self.velocity = np.array([0., 0.])
        self.trajectory = [self.position.copy()]
        self.time_elapsed = 0.0

    def apply_gravity(self):
        # Gravity applies to all missiles, but can be zero in some cases.
        return np.array([0., -self.mass * GRAVITY])

    def apply_air_resistance(self):
        # Air resistance applies to all missiles, can be overridden if the calculation differs.
        velocity_magnitude = np.linalg.norm(self.velocity)
        # Calculate the drag coefficient based on the missile's current Mach number
        dynamic_drag_coefficient = self.Cd / (1 + (self.Cd / (np.pi * self.A * self.mass)) * velocity_magnitude)
        drag_direction = self.velocity / velocity_magnitude if velocity_magnitude > 0 else np.array([0., 0.])
        return -0.5 * AIR_DENSITY * velocity_magnitude ** 2 * dynamic_drag_coefficient * self.A * drag_direction

    def apply_thrust(self):
        # This method will be specific to the subclass implementation.
        raise NotImplementedError("The apply_thrust method should be implemented by subclasses")

    def update_forces(self):
        # Update forces. This will call the relevant methods that can be overridden by subclasses.
        forces = np.array([0., 0.])
        forces += self.apply_gravity()
        forces += self.apply_air_resistance()
        forces += self.apply_thrust()
        return forces

    def update_motion(self):
        total_force = self.update_forces()
        acceleration = total_force / self.mass
        self.velocity += acceleration * TIME_STEP
        self.position += self.velocity * TIME_STEP
        self.trajectory.append(self.position.copy())
        self.time_elapsed += TIME_STEP

    def plot_trajectory(self, target):
        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Missile Trajectory')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title('Missile Trajectory')
        plt.grid(True)

        target.plot()


class BallisticMissile(Missile):
    DEFAULT_MASS = 1000
    DEFAULT_FUEL_MASS = 300
    DEFAULT_CROSS_SECTION_AREA = 0.3
    DEFAULT_DRAG_COEFFICIENT = 0.5
    DEFAULT_FUEL_CONSUMPTION_RATE = 5

    def __init__(self, target, launch_angle, max_thrust, burn_time, 
                mass=DEFAULT_MASS, fuel_mass=DEFAULT_FUEL_MASS, cross_section_area=DEFAULT_CROSS_SECTION_AREA, 
                drag_coefficient=DEFAULT_DRAG_COEFFICIENT, fuel_consumption_rate=DEFAULT_FUEL_CONSUMPTION_RATE,
                ):
        super().__init__(target, mass, cross_section_area, drag_coefficient)
        self.launch_angle = launch_angle
        self.max_thrust = max_thrust  # The maximum thrust when the missile has full fuel
        self.burn_time = burn_time
        self.fuel_mass = fuel_mass  # The initial fuel mass
        self.fuel_consumption_rate = fuel_consumption_rate  # Rate at which fuel is consumed (kg/s)
        self.propulsion_end_position = None

    def apply_thrust(self):
        angle_rad = np.radians(self.launch_angle)
        fuel_consumed = self.fuel_consumption_rate * TIME_STEP
        self.fuel_mass = max(self.fuel_mass - fuel_consumed, 0)
        self.mass = max(self.mass - fuel_consumed, self.mass - self.DEFAULT_FUEL_MASS)
        if self.time_elapsed <= self.burn_time and self.fuel_mass > 0:
            current_thrust = self.max_thrust * (self.fuel_mass / (self.mass - self.fuel_mass))
            # Calculate the propulsion force
            return np.array([current_thrust * np.cos(angle_rad), current_thrust * np.sin(angle_rad)])
        else:
            return np.array([0., 0.])

    def update_motion(self):
        if self.time_elapsed <= self.burn_time and self.fuel_mass > 0:
            self.propulsion_end_position = self.position.copy()
        super().update_motion()

    def objective(self, fuel_efficiency_weight=1.0, time_penalty_weight=1.0):
        initial_fuel_mass = self.fuel_mass
        while self.velocity[1] >= 0 or self.position[1] > self.target.position[1]:
            self.update_motion()
            if self.target.is_hit_by(self):
                return 0  # Target hit
        # If the missile has passed the target's altitude or fallen to the ground, compute the miss distance
        distance_to_target = np.linalg.norm(self.position - self.target.position)
        
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
            missile = cls(target, mass=BallisticMissile.DEFAULT_MASS, fuel_mass=BallisticMissile.DEFAULT_FUEL_MASS, cross_section_area=BallisticMissile.DEFAULT_CROSS_SECTION_AREA, drag_coefficient=BallisticMissile.DEFAULT_DRAG_COEFFICIENT, fuel_consumption_rate=BallisticMissile.DEFAULT_FUEL_CONSUMPTION_RATE,
                        launch_angle=launch_angle, max_thrust=thrust, burn_time=burn_time)
            return missile.objective()

        result = minimize(_objective, initial_guess, bounds=bounds, callback=callback)
        return result.x

    def simulate_trajectory(self):
        while not self.target.is_hit_by(self):
            self.update_motion()
            if self.position[1] < 0 and self.velocity[1] < 0:
                break

    def plot_trajectory(self, target):
        super().plot_trajectory(target)
        if self.propulsion_end_position is not None:
            plt.scatter(*self.propulsion_end_position, color='orange', s=50, label='Propulsion End', zorder=5)


class CruiseMissile(Missile):
    DEFAULT_MASS = 1000
    DEFAULT_FUEL_MASS = 300
    DEFAULT_FUEL_CONSUMPTION_RATE = 5
    DEFAULT_CROSS_SECTION_AREA = 0.3
    DEFAULT_DRAG_COEFFICIENT = 0.5
    DEFAULT_LIFT_COEFFICIENT = 0.3
    DEFAULT_MAX_FLIGHT_TIME = 3600
    TERMINAL_PHASE_START_DISTANCE = 400
    CRUISING_ALTITUDE = 1000

    def __init__(self, target, launch_angle, engine_thrust, boost_time,
                mass=DEFAULT_MASS, fuel_mass=DEFAULT_FUEL_MASS, fuel_consumption_rate=DEFAULT_FUEL_CONSUMPTION_RATE, cross_section_area=DEFAULT_CROSS_SECTION_AREA, drag_coefficient=DEFAULT_DRAG_COEFFICIENT, lift_coefficient=DEFAULT_LIFT_COEFFICIENT,
                max_flight_time=DEFAULT_MAX_FLIGHT_TIME, terminal_phase_start=TERMINAL_PHASE_START_DISTANCE, cruising_altitude=CRUISING_ALTITUDE):
        super().__init__(target, mass, cross_section_area, drag_coefficient)
        self.fuel_mass = fuel_mass
        self.lift_coefficient = lift_coefficient
        self.max_flight_time = max_flight_time
        self.launch_angle = launch_angle
        self.engine_thrust = engine_thrust
        self.boost_time = boost_time
        self.fuel_consumption_rate = fuel_consumption_rate
        self.terminal_phase_start = terminal_phase_start
        self.cruising_altitude = cruising_altitude
        self.guidance_system = 'launch'
        self.phase_switch_positions = []

    def apply_thrust(self):
        fuel_consumed = self.fuel_consumption_rate * TIME_STEP
        self.fuel_mass = max(self.fuel_mass - fuel_consumed, 0)
        self.mass = max(self.mass - fuel_consumed, self.mass - self.DEFAULT_FUEL_MASS)
        
        if self.guidance_system == 'launch':
            return self.launch_phase_thrust()
        elif self.guidance_system == 'cruise':
            return self.cruise_phase_thrust()
        elif self.guidance_system == 'terminal':
            return self.terminal_phase_thrust()
        else:
            raise ValueError("Invalid guidance system phase")

    def apply_gravity(self):
        if self.guidance_system in ['launch', 'terminal']:
            return super().apply_gravity()
        else:
            # Calculate the lift force to counter gravity at cruising altitude
            weight = self.mass * GRAVITY
            velocity_magnitude = np.linalg.norm(self.velocity)
            required_lift_coefficient = weight / (0.5 * AIR_DENSITY * velocity_magnitude ** 2 * self.A)
            
            # Adjust lift based on altitude error
            # This is a simplification, in reality, the missile would need to adjust its angle of attack and thrust to adjust the lift force
            # To make sure the vertical component of the velocity close to zero to maintain altitude
            altitude_error = self.cruising_altitude - self.position[1]
            lift_adjustment_factor = np.clip(altitude_error, -1, 1)  # More sensitive adjustment
            effective_lift_coefficient = required_lift_coefficient + lift_adjustment_factor * self.lift_coefficient

            # Calculate lift force
            lift_force = 0.5 * AIR_DENSITY * velocity_magnitude ** 2 * effective_lift_coefficient * self.A
            return super().apply_gravity() + np.array([0., lift_force])

    def update_guidance_and_navigation(self, target):
        # Switch to the cruise phase when the missile has reached the cruising altitude
        if self.position[1] >= self.cruising_altitude and self.guidance_system == 'launch':
            self.guidance_system = 'cruise'
            self.phase_switch_positions.append(self.position.copy())
        # Switch to the terminal phase when the missile is within the horizontal distance to the target
        horizontal_distance = target.position[0] - self.position[0]
        if horizontal_distance <= self.terminal_phase_start and self.guidance_system == 'cruise':
            self.guidance_system = 'terminal'
            self.phase_switch_positions.append(self.position.copy())

    def launch_phase_thrust(self):
        if self.time_elapsed <= self.boost_time and self.fuel_mass > 0:
            angle_rad = np.radians(self.launch_angle)
            thrust = np.array([self.engine_thrust * np.cos(angle_rad), self.engine_thrust * np.sin(angle_rad)])
        else:
            thrust = np.array([0., 0.])
        return thrust

    def cruise_phase_thrust(self):
        # Maintain thrust only in the horizontal direction during the cruise phase
        if self.fuel_mass > 0:
            return np.array([self.engine_thrust, 0.])
        else:
            return np.array([0., 0.])

    def terminal_phase_thrust(self):
        # Calculate the horizontal and vertical distance to the target
        horizontal_distance = self.target.position[0] - self.position[0]
        vertical_distance = self.position[1] - self.target.position[1]

        # Adjust descent angle based on proximity to the target
        steepness_factor = np.clip(1 - horizontal_distance / self.terminal_phase_start, 0, 1)
        descent_angle_rad = np.arctan(steepness_factor * (vertical_distance / horizontal_distance))

        # Calculate desired velocity in the direction of the descent angle
        desired_velocity = np.array([
            np.cos(descent_angle_rad) * np.linalg.norm(self.velocity),
            -np.sin(descent_angle_rad) * np.linalg.norm(self.velocity)
        ])

        # Calculate the difference between the desired and current velocity
        velocity_difference = desired_velocity - self.velocity

        # Calculate the acceleration needed to match the desired velocity
        required_acceleration = velocity_difference / TIME_STEP

        # Limit the acceleration
        required_acceleration = np.clip(required_acceleration, -self.engine_thrust, self.engine_thrust)

        # Calculate the thrust needed to achieve the required acceleration
        thrust = required_acceleration * self.mass

        return thrust

    def update_motion(self, target):
        self.update_guidance_and_navigation(target)
        super().update_motion()

    def objective(self):
        # Simulate the launch phase
        while self.time_elapsed <= self.boost_time:
            self.update_motion(self.target)
        # Evaluate the objective function: minimize the difference between current altitude and cruising altitude
        return abs(self.position[1] - self.cruising_altitude)

    @classmethod
    def optimize_launch_phase(cls, target, initial_guess, bounds):
        def _callback(params):
            print(f"Current iteration parameters: launch angle = {params[0]}, thrust = {params[1]}, boost time = {params[2]}")

        def _objective(params):
            launch_angle, engine_thrust, boost_time = params
            missile = cls(target, mass=CruiseMissile.DEFAULT_MASS, fuel_mass=CruiseMissile.DEFAULT_FUEL_MASS, fuel_consumption_rate=CruiseMissile.DEFAULT_FUEL_CONSUMPTION_RATE, cross_section_area=CruiseMissile.DEFAULT_CROSS_SECTION_AREA, drag_coefficient=CruiseMissile.DEFAULT_DRAG_COEFFICIENT, lift_coefficient=CruiseMissile.DEFAULT_LIFT_COEFFICIENT,
                        launch_angle=launch_angle, engine_thrust=engine_thrust, boost_time=boost_time)
            return missile.objective()

        result = minimize(_objective, initial_guess, bounds=bounds, callback=_callback)
        return result.x

    def simulate_trajectory(self):
        while not self.target.is_hit_by(self):
            self.update_motion(self.target)
            if self.time_elapsed >= self.max_flight_time:
                break

    def plot_trajectory(self, target):
        super().plot_trajectory(target)
        for pos in self.phase_switch_positions:
            plt.scatter(*pos, color='blue', s=50, zorder=5, label='Phase Switch')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())


class BoostGlideVehicle(Missile):
    DEFAULT_GLIDING_ALTITUDE = 1000
    DEFAULT_BOOSTER_MASS = 1000
    DEFAULT_GLIDER_MASS = 500
    DEFAULT_BOOST_TIME = 10
    DEFAULT_GLIDE_TIME = 3600
    DEFAULT_TERMINAL_PHASE_START = 1000
    DEFAULT_DRAG_COEFFICIENT = 0.5
    DEFAULT_CROSS_SECTION_AREA = 0.3

    def __init__(self, target, launch_angle, boost_thrust, glide_lift_coefficient,
                glide_drag_coefficient=DEFAULT_DRAG_COEFFICIENT, gliding_altitude=DEFAULT_GLIDING_ALTITUDE,
                boost_time=DEFAULT_BOOST_TIME, glide_time=DEFAULT_GLIDE_TIME, terminal_phase_start=DEFAULT_TERMINAL_PHASE_START,
                booster_mass=DEFAULT_BOOSTER_MASS, glider_mass=DEFAULT_GLIDER_MASS):
        super().__init__(target, booster_mass + glider_mass, self.DEFAULT_CROSS_SECTION_AREA, glide_drag_coefficient)
        self.launch_angle = launch_angle
        self.boost_thrust = boost_thrust
        self.gliding_altitude = gliding_altitude
        self.glide_lift_coefficient = glide_lift_coefficient
        self.boost_time = boost_time
        self.glide_time = glide_time
        self.terminal_phase_start = terminal_phase_start
        self.booster_mass = booster_mass
        self.glider_mass = glider_mass
        self.phase = "boost"
        self.phase_switch_positions = []

    def apply_thrust(self):
        if self.phase == "boost":
            angle_rad = np.radians(self.launch_angle)
            return np.array([self.boost_thrust * np.cos(angle_rad), self.boost_thrust * np.sin(angle_rad)])
        elif self.phase == "terminal":
            horizontal_distance = self.target.position[0] - self.position[0]
            vertical_distance = self.position[1] - self.target.position[1]
            steepness_factor = np.clip(1 - horizontal_distance / self.terminal_phase_start, 0, 1)
            descent_angle_rad = np.arctan(steepness_factor * (vertical_distance / horizontal_distance))
            desired_velocity = np.array([
                np.cos(descent_angle_rad) * np.linalg.norm(self.velocity),
                -np.sin(descent_angle_rad) * np.linalg.norm(self.velocity)
            ])
            velocity_difference = desired_velocity - self.velocity
            required_acceleration = velocity_difference / TIME_STEP
            required_acceleration = np.clip(required_acceleration, -self.boost_thrust, self.boost_thrust)
            return required_acceleration * self.mass
        else:
            return np.array([0., 0.])

    def apply_gravity(self):
        if self.phase in ["boost", "terminal"]:
            return super().apply_gravity()
        else:  # Glide phase
            velocity_magnitude = np.linalg.norm(self.velocity)
            lift_force = 0.5 * AIR_DENSITY * velocity_magnitude ** 2 * self.glide_lift_coefficient * self.A
            return super().apply_gravity() + np.array([0., lift_force])

    def update_guidance_and_navigation(self):
        if self.phase == "boost" and self.position[1] >= self.gliding_altitude:
            self.phase = "glide"
            self.mass = self.glider_mass  # Discard the booster mass after the boost phase
            self.phase_switch_positions.append(self.position.copy())
        elif self.phase == "glide" and (self.target.position[0] - self.position[0]) <= self.terminal_phase_start:
            self.phase = "terminal"
            self.phase_switch_positions.append(self.position.copy())
            # Implement any specific logic for the terminal phase here

    def update_motion(self):
        self.update_guidance_and_navigation()
        super().update_motion()

    def objective(self):
        distance_to_glide_altitude = abs(self.gliding_altitude - self.position[1])
        while not self.target.is_hit_by(self) and self.phase == "boost":
            self.update_motion()
            if self.position[1] < 0 or self.position[0] > self.target.position[0]:
                break
            distance_to_glide_altitude += abs(self.gliding_altitude - self.position[1])
        distance_to_target_altitude = abs(self.target.position[1] - self.position[1])
        while not self.target.is_hit_by(self) and self.phase == "glide":
            self.update_motion()
            if self.position[1] < 0 or self.position[0] > self.target.position[0]:
                break
            distance_to_target_altitude += abs(self.target.position[1] - self.position[1])
        while not self.target.is_hit_by(self) and self.phase == "terminal":
            self.update_motion()
            if self.position[1] < 0 or self.position[0] > self.target.position[0]:
                break
        distance_to_target = np.linalg.norm(self.position - self.target.position)
        return distance_to_glide_altitude + distance_to_target

    @classmethod
    def optimize_launch_glide_phases(cls, target, initial_guess, bounds):
        def callback(params):
            print(f"Current iteration parameters: launch angle = {params[0]}, thrust = {params[1]}, glide lift coefficient = {params[2]}")
        
        def objective(params):
            launch_angle, boost_thrust, glide_lift_coefficient = params
            vehicle = cls(target, launch_angle=launch_angle, boost_thrust=boost_thrust, glide_lift_coefficient=glide_lift_coefficient,
                        glide_drag_coefficient=BoostGlideVehicle.DEFAULT_DRAG_COEFFICIENT, gliding_altitude=BoostGlideVehicle.DEFAULT_GLIDING_ALTITUDE,
                        booster_mass=BoostGlideVehicle.DEFAULT_BOOSTER_MASS, glider_mass=BoostGlideVehicle.DEFAULT_GLIDER_MASS,
                        boost_time=BoostGlideVehicle.DEFAULT_BOOST_TIME, glide_time=BoostGlideVehicle.DEFAULT_GLIDE_TIME)
            return vehicle.objective()

        result = minimize(objective, initial_guess, bounds=bounds, callback=callback)
        return result.x

    def simulate_trajectory(self):
        while not self.target.is_hit_by(self):
            self.update_motion()
            if self.position[1] < 0 or self.position[0] > self.target.position[0]:
                break

    def plot_trajectory(self, target):
        super().plot_trajectory(target)
        for pos in self.phase_switch_positions:
            plt.scatter(*pos, color='blue', s=50, zorder=5, label='Phase Switch')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())


class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = float(np.linalg.norm(missile.position - self.position))
        return distance_to_missile < HIT_RADIUS

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')


def optimize_ballistic_missile():
    # Ballistic Missile Optimization
    ballistic_target_position = [10000, 1000]
    ballistic_target = Target(ballistic_target_position)
    initial_guess = [45, 200000, 10]
    bounds = [(0, 90), (0, 500000), (0, 60)]
    
    optimal_launch_angle, optimal_thrust, optimal_burn_time = BallisticMissile.find_optimal_conditions(
        ballistic_target, initial_guess, bounds)
    
    ballistic_missile = BallisticMissile(
        target=ballistic_target, launch_angle=optimal_launch_angle, 
        max_thrust=optimal_thrust, burn_time=optimal_burn_time)
    
    ballistic_missile.simulate_trajectory()
    ballistic_missile.plot_trajectory(ballistic_target)

def optimize_cruise_missile():
    # Cruise Missile Optimization
    cruise_target_position = [10000, 100]
    cruise_target = Target(cruise_target_position)
    initial_guess = [45, 50000, 10]
    bounds = [(0, 90), (10000, 100000), (5, 20)]
    
    optimal_launch_angle, optimal_engine_thrust, optimal_boost_time = CruiseMissile.optimize_launch_phase(
        cruise_target, initial_guess, bounds)
    
    cruise_missile = CruiseMissile(
        target=cruise_target, launch_angle=optimal_launch_angle, 
        engine_thrust=optimal_engine_thrust, boost_time=optimal_boost_time)
    
    cruise_missile.simulate_trajectory()
    cruise_missile.plot_trajectory(cruise_target)

def optimize_boost_glide_vehicle():
    # Boost Glide Vehicle
    boost_glide_target_position = [10000, 200]
    boost_glide_target = Target(boost_glide_target_position)
    initial_guess = [45, 100000, 0.2]
    bounds = [(0, 90), (10000, 1000000), (0.1, 0.3)]
    optimal_launch_angle, optimal_boost_thrust, optimal_glide_lift_coefficient = BoostGlideVehicle.optimize_launch_glide_phases(
        boost_glide_target, initial_guess, bounds)

    boost_glide_vehicle = BoostGlideVehicle(
        target=boost_glide_target, launch_angle=optimal_launch_angle, boost_thrust=optimal_boost_thrust, glide_lift_coefficient=optimal_glide_lift_coefficient)

    boost_glide_vehicle.simulate_trajectory()
    boost_glide_vehicle.plot_trajectory(boost_glide_target)

def main():
    #optimize_ballistic_missile()
    #optimize_cruise_missile()
    optimize_boost_glide_vehicle()

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# The desirable trajectory for a boost-glide vehicle involves three distinct phases:
# A boost phase where the missile is launched and gains altitude quickly.
# A glide phase where the missile levels out and glides at a constant altitude, potentially with some slight descent to maintain speed and extend range.
# A terminal phase where the missile descends rapidly to the target.


# May change the angle of attack to adjust the lift force to maintain altitude.
# https://chat.openai.com/c/77161094-a1d1-46d3-8062-3cf37e08689e