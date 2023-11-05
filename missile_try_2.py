import matplotlib.pyplot as plt
import numpy as np

# Constants
G = 9.81  # Gravitational acceleration (m/s^2)
RHO = 1.225  # Air density at sea level (kg/m^3)
DT = 0.01  # Time step (s)

class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < 10

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')
        plt.legend()

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
                climb_angle_rad = np.deg2rad(45)  # Example climb angle of 45 degrees
                return self.engine_thrust * np.array([np.cos(climb_angle_rad), np.sin(climb_angle_rad)])
            else:
                # Transition to cruise phase after boost
                self.guidance_system = 'cruise'
                return np.array([self.engine_thrust, 0])
        elif self.guidance_system == 'cruise':
            # Maintain the propulsion in the horizontal direction
            return np.array([self.engine_thrust, 0])
        elif self.guidance_system == 'terminal':
            # No propulsion in terminal phase, missile is on final approach
            return np.array([0, 0])

    def apply_gravity(self):
        if self.guidance_system == 'launch' and self.position[1] < self.cruising_altitude:
            return np.array([0, -self.mass * G])
        else:
            # Gravity is countered by lift in cruise and terminal phases
            return np.array([0, 0])

    def apply_air_resistance(self):
        velocity_magnitude = np.linalg.norm(self.velocity)
        drag_direction = self.velocity / velocity_magnitude if velocity_magnitude > 0 else np.array([0., 0.])
        return -0.5 * RHO * velocity_magnitude ** 2 * self.Cd * self.A * drag_direction

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

        drag_force = self.apply_air_resistance()
        thrust_force = self.apply_propulsion()

        # Since gravity is countered by lift, the only forces are drag and thrust
        total_force = drag_force + thrust_force
        acceleration = total_force / self.mass

        self.velocity += acceleration * DT
        self.position += self.velocity * DT
        self.trajectory.append(self.position.copy())
        self.time_elapsed += DT

def main():
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

if __name__ == "__main__":
    main()