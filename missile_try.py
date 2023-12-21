import math

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
            missile = cls(target, mass=1000, fuel_mass=300, cross_section_area=0.3, drag_coefficient=0.5, 
                        launch_angle=launch_angle, max_thrust=thrust, burn_time=burn_time, 
                        fuel_consumption_rate=5)
            return missile.objective(target)

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

    @classmethod
    def optimize_launch_phase(cls, target, initial_guess, bounds):
        def _callback(params):
            print(f"Current iteration parameters: launch angle = {params[0]}, thrust = {params[1]}, boost time = {params[2]}")

        def _objective(params):
            launch_angle, engine_thrust, boost_time = params
            missile = cls(target, mass=1000, cross_section_area=0.3, drag_coefficient=0.5, max_flight_time=9600, launch_angle=launch_angle, engine_thrust=engine_thrust, boost_time=boost_time)
            # Simulate the launch phase
            while missile.time_elapsed <= boost_time:
                missile.update_motion(target)
            # Evaluate the objective function: minimize the difference between current altitude and cruising altitude
            altitude_difference = abs(missile.position[1] - missile.cruising_altitude)
            return altitude_difference

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


class Target:
    def __init__(self, position):
        self.position = np.array(position)

    def is_hit_by(self, missile):
        distance_to_missile = np.linalg.norm(missile.position - self.position)
        return distance_to_missile < HIT_RADIUS

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='red', marker='X', s=100, label='Target')


def main():
    # Target positions
    ballistic_target_position = [10000, 1000]  # Target for ballistic missile
    cruise_target_position = [10000, 100]  # Target for cruise missile

    # Ballistic Missile Optimization
    ballistic_target = Target(ballistic_target_position)
    initial_guess = [45, 200000, 10]  # Initial guess for the optimization
    bounds = [(0, 90), (0, 500000), (0, 60)]  # Bounds for launch_angle, thrust, and burn_time
    optimal_launch_angle, optimal_thrust, optimal_burn_time = BallisticMissile.find_optimal_conditions(ballistic_target, initial_guess, bounds)

    # Create a ballistic missile instance with the optimal conditions found
    ballistic_missile = BallisticMissile(target=ballistic_target,
        launch_angle=optimal_launch_angle, max_thrust=optimal_thrust, burn_time=optimal_burn_time
    )

    ballistic_missile.simulate_trajectory()
    ballistic_missile.plot_trajectory(ballistic_target)

    # Cruise Missile Optimization
    cruise_target = Target(cruise_target_position)
    initial_guess = [45, 50000, 10]  # Initial guess: launch angle, engine thrust, boost time
    bounds = [(0, 90), (10000, 100000), (5, 20)]  # Bounds for the parameters
    optimal_launch_angle, optimal_engine_thrust, optimal_boost_time = CruiseMissile.optimize_launch_phase(cruise_target, initial_guess, bounds)

    # Create the cruise missile
    cruise_missile = CruiseMissile(target=cruise_target,
        launch_angle=optimal_launch_angle, engine_thrust=optimal_engine_thrust, boost_time=optimal_boost_time
    )

    cruise_missile.simulate_trajectory()
    cruise_missile.plot_trajectory(cruise_target)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

"""
Apply proportional navigation guidance to adjust the missile's trajectory based on the rate of change of the line-of-sight angle to the target.

Please adjust so that the lift force is applied on all phases of cruise missile, but adjusted according to angle of attack and other factors.

Fuel Efficiency Objective: If you want to optimize fuel efficiency for the cruise missile as well, you might need to add a similar objective function as in the BallisticMissile for the optimization process.

Please summarize all the forces act on the missile and improve the realism of the simulation.

Dynamic Target:
The target can be made to move, which can be more realistic.

Logging and Verbosity:
Add logging capabilities to help with debugging or understanding the missile's behavior over time.

Physical Constraints:
Ensure that physical constraints are applied. For example, the missile should not be able to accelerate beyond the maximum thrust or decelerate beyond the maximum drag force. Also, the missile should not be able to turn faster than its maximum turn rate.

Configuration via External Files:
Allow for the missile and simulation parameters to be loaded from an external configuration file, making it easier to run different scenarios without changing the code.

Create a base Physics class that both missile types can inherit from. This class will handle the basic physics such as gravity, air resistance, and propulsion.

Use properties for attributes where dynamic calculation is needed instead of methods.

Create a GuidanceSystem class that handles the logic of guiding the missile, which can be inherited and extended by specific missile types.
A GuidanceSystem class with methods for updating missile heading and position. This modularity allows for different guidance strategies to be implemented and tested with the CruiseMissile class.

"""

"""
總體而言，由彈道飛彈發射的「機動重返載具」（MaRV）在許多情況下，性能都優於高超音速武器，美國國會預算辦公室（CBO）最近一項分析預測，其成本將比高超音速武器便宜3分之1。

錢學森彈道
Sanger彈道
"""

"""
https://indsr.org.tw/respublicationcon?uid=12&resid=705&pid=2588&typeid=3
1.20 u@s.RX 04/23 fBg:/ 复制打开抖音，看看【娱小扒的作品】狼群特有的行进站位，感觉它们好有谋略 # 狼群 #... https://v.douyin.com/iRrP8wnk/
https://academic-accelerator.com/encyclopedia/zh/list-of-military-tactics
https://zh.wikipedia.org/zh-hant/%E5%86%9B%E4%BA%8B%E6%88%98%E6%9C%AF%E5%88%97%E8%A1%A8
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

"""
https://github.com/rafal9820/9M723-Iskander-missile-trajectory

https://github.com/topics/rocket-simulator

https://github.com/topics/missile-defence-simulation

https://github.com/topics/missile-simulation

https://github.com/topics/missile

https://github.com/ferencdv/missile_trajectory_simulator

https://github.com/CoopLo/am_205_project
"""
