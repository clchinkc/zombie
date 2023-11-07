import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

GRAVITY = 9.81  # Gravitational acceleration (m/s^2)
AIR_DENSITY = 1.225  # Air density at sea level (kg/m^3)
TIME_STEP = 0.01  # Time step (s)
HIT_RADIUS = 10  # Radius within which the target is considered hit (m)

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
        return np.array([0., -self.mass * GRAVITY])

    def apply_air_resistance(self):
        # Air resistance applies to all missiles, can be overridden if the calculation differs.
        velocity_magnitude = np.linalg.norm(self.velocity)
        drag_direction = self.velocity / velocity_magnitude if velocity_magnitude > 0 else np.array([0., 0.])
        return -0.5 * AIR_DENSITY * velocity_magnitude ** 2 * self.Cd * self.A * drag_direction

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

    def apply_thrust(self):
        angle_rad = np.radians(self.launch_angle)
        if self.time_elapsed <= self.burn_time and self.fuel_mass > 0:
            current_thrust = self.max_thrust * (self.fuel_mass / (self.mass - self.fuel_mass))
            # Decrease fuel mass
            self.fuel_mass -= self.fuel_consumption_rate * TIME_STEP
            # Ensure fuel mass doesn't become negative
            self.fuel_mass = max(self.fuel_mass, 0)
            # Calculate the propulsion force
            return np.array([current_thrust * np.cos(angle_rad), current_thrust * np.sin(angle_rad)])
        else:
            return np.array([0., 0.])

    def update_motion(self):
        self.mass -= self.fuel_consumption_rate * TIME_STEP
        self.mass = max(self.mass, 0)
        super().update_motion()

    def objective(self, target, fuel_efficiency_weight=1.0, time_penalty_weight=1.0):
        initial_fuel_mass = self.fuel_mass
        while self.velocity[1] >= 0 or self.position[1] > target.position[1]:
            self.update_motion()
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
        self.launch_angle = 45  # Launch angle in degrees
        self.guidance_system = 'launch'  # Start with launch phase guidance

    def apply_thrust(self):
        if self.guidance_system == 'launch':
            if self.time_elapsed <= self.boost_time:
                # Calculate the required climb angle to reach cruising altitude at the end of the boost phase
                # Assume that the initial horizontal velocity is the same as the initial vertical velocity
                avg_vertical_velocity = self.cruising_altitude / self.boost_time
                self.launch_angle = np.degrees(np.arctan(avg_vertical_velocity / avg_vertical_velocity))
                
                climb_angle_rad = np.radians(self.launch_angle)
                vertical_thrust_component = self.engine_thrust * np.sin(climb_angle_rad)
                horizontal_thrust_component = self.engine_thrust * np.cos(climb_angle_rad)

                return np.array([
                    horizontal_thrust_component + self.apply_air_resistance()[0],
                    vertical_thrust_component + self.mass * GRAVITY + self.apply_air_resistance()[1]
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

    def update_guidance_and_navigation(self, target):
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

    def update_motion(self, target):
        self.update_guidance_and_navigation(target)
        super().update_motion()


class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < HIT_RADIUS

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
        missile.update_motion()

    missile.plot_trajectory(target)
    
    target_position = [10000, 100]  # Assume a flat terrain and target at sea level
    target = Target(target_position)

    # Create the target and cruise missile
    cruise_missile = CruiseMissile(mass=1000, cross_section_area=0.3, drag_coefficient=0.5, engine_thrust=5000, max_flight_time=9600)
    
    # Update and plot the cruise missile's trajectory
    while cruise_missile.time_elapsed < cruise_missile.max_flight_time:
        cruise_missile.update_motion(target)
        if target.is_hit_by(cruise_missile):
            break

    cruise_missile.plot_trajectory(target)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

"""
Please modify the method so that the cruise missile climbs to cruise altitude exactly at the end of the boost phase instead of climbing 45 degrees in the boost phase and raising vertically to the cruise altitude in the cruise phase.
Introduce constants at the class level to replace magic numbers.

Dynamic Time Step:
A fixed time step (DT) works for simple simulations, but for a more accurate simulation, especially when the missile's speed changes drastically, an adaptive time-stepping method could be used.

Logging and Verbosity:
Add logging capabilities to help with debugging or understanding the missile's behavior over time.

Physical Constraints:
Ensure that physical constraints are applied, such as non-negative mass and preventing the fuel mass from increasing during flight.

Error Handling:
Add error handling to manage potential issues such as division by zero or optimization failures.

Configuration via External Files:
Allow for the missile and simulation parameters to be loaded from an external configuration file, making it easier to run different scenarios without changing the code.

Modularize Code:
Separate the classes and the main execution logic into different files/modules for better organization.

Create a base Physics class that both missile types can inherit from. This class will handle the basic physics such as gravity, air resistance, and propulsion.

Use properties for attributes where dynamic calculation is needed instead of methods.

Create a GuidanceSystem class that handles the logic of guiding the missile, which can be inherited and extended by specific missile types.
A GuidanceSystem class with methods for updating missile heading and position. This modularity allows for different guidance strategies to be implemented and tested with the CruiseMissile class.

"""

"""
Let’s synthesize a comprehensive overview of missile technology development over time by incorporating the previously provided information:

**Early Missile Development:**

- **Cruise Missiles:**
  Originally developed during World War II, cruise missiles have been integral to military strategies due to their versatility. They can be launched from land, sea, air, or subsea platforms and are known for their precision. These missiles evolved from earlier, slower versions to modern variants that can achieve high subsonic to hypersonic speeds. They are guided by a variety of onboard systems and can be remotely controlled.

- **Ballistic Missiles:**
  Ballistic missiles also have their roots in World War II and are known for their long-range capabilities. These missiles are launched directly into space on a rocket, eventually following a ballistic (arc-like) trajectory back into the Earth's atmosphere at hypersonic speeds to reach their target. They are traditionally categorized by their range: short, medium, intermediate, and intercontinental (ICBMs).

**Mid-Century Advancements:**

- **Maneuverable Re-entry Vehicles (MaRVs) and Multiple Independently targetable Re-entry Vehicles (MIRVs):**
  To improve the efficacy of ballistic missiles against missile defense systems, MaRVs and MIRVs were introduced. These technologies allowed for in-flight maneuverability and the ability to hit multiple targets with a single launch, respectively.

- **Boost-Glide Vehicles:**
  Combining aspects of ballistic and cruise missile technologies, boost-glide vehicles are launched into space on a rocket before transitioning to a glide phase in the atmosphere, allowing for long-range, high-speed strikes with greater maneuverability than traditional ballistic missiles.

**Hypersonic Era and Modern Developments:**

- **Hypersonic Cruise Missiles:**
  These missiles are at the forefront of modern missile technology. Using scramjet engines, they are capable of flight at the edge of the atmosphere (mesosphere) at hypersonic speeds. Their ability to maneuver at lower altitudes than ballistic missiles makes them hard to detect and intercept, providing a strategic advantage.

- **Hypersonic Glide Vehicles (HGVs):**
  Launched from existing missile systems, HGVs detach from their boosters and glide at hypersonic speeds to their targets. Their maneuverability and speed make them formidable systems that can potentially evade current missile defense mechanisms.

- **Intercontinental Ballistic Missile (ICBM):**
  Representing the pinnacle of traditional ballistic missile technology, ICBMs can deliver a payload over intercontinental distances. While they can be detected due to their large launch signature, their speed and trajectory make interception a significant challenge.

The trajectory of missile technology highlights a continuous drive toward faster, more versatile, and harder-to-detect systems. From the jet-engine powered cruise missiles, which brought precision to missile attacks, to the ICBMs that became symbols of strategic deterrence, and now to the hypersonic missiles that are redefining the pace and method of future warfare, each development encapsulates the advancing military technology frontiers. The shift to hypersonic speeds, in particular, is a game-changing evolution, emphasizing the critical role of speed and stealth in overcoming missile defenses and ensuring payload delivery.
"""

"""
A Cruise Missile is a guided missile that flies with constant speed to deliver a warhead at specified target over long distance with high accuracy. A Ballistic Missile is lift off directly into the high layers of the earth’s atmosphere.

Cruise Missiles
Cruise missiles are unmanned vehicles that are propelled by jet engines, much like an airplane. They can be launched from ground, air, or sea platforms. Cruise missiles remain within the atmosphere for the duration of their flight and can fly as low as a few meters off the ground. Flying low to the surface of the earth expends more fuel but makes a cruise missile very difficult to detect. Cruise missiles are self-guided and use multiple methods to accurately deliver their payload, including terrain mapping, global positioning systems (GPS) and inertial guidance, which uses motion sensors and gyroscopes to keep the missile on a pre-programmed flight path. As advanced cruise missiles approach their target, remote operators can use a camera in the nose of the missile to see what the missile sees. This gives them the option to manually guide the missile to its target or to abort the strike.

A cruise missile is a guided missile (target has to be pre-set) used against terrestrial targets.
It remains in the atmosphere throughout its flight.
It flies the major portion of its flight path at approximately constant speed.
Cruise missiles are designed to deliver a large warhead over long distances with high precision.
Modern cruise missiles are capable of travelling at supersonic or high subsonic speeds, are self-navigating, and are able to fly on a non-ballistic, extremely low-altitude trajectory.
Types of cruise missiles based on speed
Hypersonic (Mach 5): These missiles would travel at least five times the speed of sound (Mach 5). E.g. BrahMos-II.
Supersonic (Mach 2-3): These missiles travel faster than the speed of sound. E.g. BrahMos.
Subsonic (Mach 0.8): These missiles travel slower than the speed of sound. E.g. Nirbhay.
Ballistic Missile
Ballistic missiles are powered initially by a rocket or series of rockets in stages, but then follow an unpowered trajectory that arches upwards before descending to reach its intended target. Ballistic missiles can carry either nuclear or conventional warheads. There are four general classifications of ballistic missiles based on their range, or the maximum distance the missile can travel:

A ballistic missile follows a ballistic trajectory to deliver one or more warheads on a predetermined target.
A ballistic trajectory is the path of an object that is lift of but has no active propulsion during its actual flight (these weapons are guided only during relatively brief periods of flight).
Consequently, the trajectory is fully determined by a given initial velocity, effects of gravity, air resistance, and motion of the earth (Coriolis Force).
Shorter range ballistic missiles stay within the Earth’s atmosphere.
Long-range intercontinental ballistic missiles (ICBMs), are lift off on a sub-orbital flight trajectory and spend most of their flight out of the atmosphere.
Types of ballistic missiles based on the range
Short-range (tactical) ballistic missile (SRBM): Range between 300 km and 1,000 km.
Medium-range (theatre) ballistic missile (MRBM): 1,000 km to 3,500 km.
Intermediate-range (Long-Range) ballistic missile (IRBM or LRBM): 3,500 km and 5,500 km.
Intercontinental ballistic missile (ICBM): 5,500 km +
How Cruise Missiles different from Ballistic Missiles?
Cruise missiles are unmanned vehicles that are propelled by jet engines, much like an airplane. They can be lift off from ground, air, sea or submarine platforms. Cruise missiles remain within the atmosphere for the duration of their flight and can fly as low as a few meters off the ground. Flying low to the surface of the earth expends more fuel but makes a cruise missile very difficult to detect. Cruise missiles are self-guided and use multiple methods to accurately deliver their payload, including terrain mapping, global positioning systems (GPS) and inertial guidance.
"""

"""
Quasi-Ballistic missiles are ballistic missiles with a changeable trajectory, they can make unpredictable movements, and readjust themselves, even modify their ballistic path. But at the cost of range.
"""

"""
A boost glide vehicle is a type of hypersonic weapon system that combines the characteristics of both ballistic missiles and gliders. It is designed to deliver a warhead or payload to a distant target with high precision and at speeds greater than Mach 5, which is five times the speed of sound.

Here's how a boost glide system typically works:

1. **Boost Phase**: The vehicle is launched by a rocket booster, similar to a traditional ballistic missile. It ascends through the atmosphere and follows a parabolic trajectory outside of the atmosphere, or near the edge of space.

2. **Glide Phase**: Once it reaches a certain altitude and speed, the glide vehicle (the payload part) separates from the booster. Instead of following a purely ballistic trajectory that would bring it back down under the force of gravity alone, the glide vehicle enters a long-duration, high-speed glide phase in the upper atmosphere.

3. **Hypersonic Glide**: During the glide phase, the vehicle uses aerodynamic lift to extend its range and maneuver. This is the key difference from a traditional ballistic missile. The lift allows the vehicle to glide at hypersonic speeds for a prolonged period, which can be thousands of kilometers. This makes the flight path less predictable, and thus, the vehicle can potentially evade missile defense systems.

4. **Terminal Phase**: Eventually, the vehicle transitions to the terminal phase, where it descends towards its target at hypersonic speeds. This phase is characterized by high maneuverability, making it difficult to intercept.

The boost glide vehicle's ability to fly at the edge of the atmosphere allows it to achieve great speeds while also providing some stealth benefits, as it can travel at altitudes that are difficult to cover by radar systems. Additionally, the hypersonic glide phase gives it an unpredictable flight path, unlike the predictable arc of a traditional intercontinental ballistic missile (ICBM), complicating the task of defense systems designed to intercept incoming threats.
"""

"""
A guided missile is a type of missile that has the capability to adjust its flight path during flight, typically using onboard control systems. This allows it to be directed towards a predetermined target, with a high degree of accuracy. There are several key components and systems involved in the functioning of a guided missile:

1. **Propulsion System:**
   - This includes the engine and the fuel system which propel the missile. The propulsion could be a rocket for long-range and high-speed missiles, or a jet engine for some types of cruise missiles.

2. **Guidance System:**
   - This is the "brain" of the missile. The guidance system can include onboard computers, sensors, and navigation systems. It processes data and sends commands to the control surfaces to adjust the missile's flight path.

3. **Control System:**
   - This includes the fins, canards, or other control surfaces that can be adjusted to steer the missile. In some cases, thrust vectoring is used, where the direction of the propulsion can be altered.

4. **Warhead:**
   - This is the payload that the missile delivers to the target, which can be conventional explosives, nuclear, chemical, or biological agents. Some missiles may carry multiple warheads or submunitions.

5. **Targeting System:**
   - Before or during flight, a missile must be provided with information about its target. This can be done before launch (pre-set targets) or during flight (target updates via data link or onboard sensors).

6. **Flight Path:**
   - The trajectory that the missile takes can be ballistic (a high arcing path), cruise (a low-level, airplane-like path), or some combination thereof.

Guided missiles can be categorized based on various criteria:

- **Launch Platform:** Air-to-air, surface-to-air, air-to-surface, surface-to-surface, undersea-launched, etc.
- **Range:** Short-range, medium-range, long-range (intercontinental ballistic missiles).
- **Guidance Type:** Inertial, command, beam riding, terrain following, homing (active, semi-active, or passive), or GPS-guided.

The guidance and control of a missile involve complex processes, including signal processing, fluid dynamics, and physics, as well as advanced algorithms to process inputs and adjust the missile's flight path accordingly.

Would you like to know more details about a specific aspect of guided missiles, such as their history, technological development, different types, or how they are used in defense systems?
"""

"""
The guidance system of a missile is critical for its ability to accurately reach its target. It encompasses the sensors, computers, and algorithms that determine the missile's flight path and make adjustments as needed. The guidance system typically operates in four phases: launch, boost, midcourse, and terminal. Each phase may use different guidance techniques depending on the type of missile and its intended mission.

Here are some of the most common guidance methods used in missile systems:

1. **Inertial Guidance:**
   - This system uses gyroscopes and accelerometers to measure the missile's acceleration and angular velocity. By integrating these measurements over time, the missile's computer can calculate its current velocity and position relative to its starting point. Inertial guidance is self-contained and does not depend on external signals, making it immune to jamming, but it can drift over time and become less accurate.

2. **Command Guidance:**
   - This involves the missile being guided by commands from an external source, such as a ground control station. The missile has a receiver that picks up signals from the control station, which tracks the missile and the target, and sends corrections to the missile's control system.

3. **Beam Riding:**
   - The missile is guided by staying within a beam of energy (like a laser or radar) that is pointed at the target. The missile detects the edges of the beam and keeps itself centered within the beam as it flies.

4. **Homing Guidance:**
   - This can be active, passive, or semi-active:
     - **Active Homing:** The missile carries its own radar or sonar system to locate the target and adjusts its path based on the signals that it receives.
     - **Semi-Active Homing:** The missile uses a signal from an external source that is reflected off the target, like a ground station's radar illuminating the target and the missile homing in on the reflected signal.
     - **Passive Homing:** The missile targets the signature of the target itself, such as heat (infrared) or electromagnetic emissions.

5. **Terrestrial Navigation:**
   - This system involves comparing the terrain over which the missile is flying with a stored map to adjust its flight path. This type of guidance is often used by cruise missiles.

6. **Satellite Navigation:**
   - With the availability of global positioning systems (GPS), missiles can now use satellite signals to determine their position and velocity and adjust their course accordingly. GPS guidance is commonly combined with inertial navigation to improve accuracy and reliability.

7. **Astro-inertial Guidance:**
   - Similar to inertial guidance, but includes celestial navigation as a reference to correct inertial drift, typically used in long-range intercontinental ballistic missiles (ICBMs).

Each guidance method has its own advantages and disadvantages in terms of complexity, cost, susceptibility to countermeasures, and accuracy. Modern missiles often use a combination of these guidance methods to improve accuracy and resistance to jamming or other forms of interference. For example, a missile might use inertial guidance to get close to its target and then switch to active or semi-active homing for final approach and impact.
"""
