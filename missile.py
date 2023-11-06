import math

import pygame

# Initialize Pygame
pygame.init()

# Define Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FPS = 60
TIME_PER_FRAME = 1.0 / FPS  # Time per frame in seconds
GRAVITY = 9.8  # Gravity constant
AIR_RESISTANCE_COEFF = 0.01  # Air resistance coefficient for linear drag

# Setup the Pygame Window
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Missile Simulation with Physics, Air Resistance, and Target")

class Missile:
    def __init__(self, propulsion_force, weight, target_distance):
        self.x = 50
        self.y = HEIGHT
        self.propulsion_force = propulsion_force
        self.weight = weight
        self.target_distance = target_distance
        self.has_fuel = True
        
        # Assuming some method to calculate initial conditions
        self.velocity, self.angle = self.calculate_initial_conditions()

        self.velocity_x = self.velocity * math.cos(math.radians(self.angle))
        self.velocity_y = self.velocity * math.sin(math.radians(self.angle))

        # Initialize accelerations to zero
        self.acceleration_x = 0
        self.acceleration_y = 0

    def calculate_initial_conditions(self):
        # Step 1: Calculate Time of Flight
        # For horizontal motion: distance = velocity * time
        # Solve for time: time = distance / velocity
        time_of_flight = self.target_distance / (self.propulsion_force * math.sqrt(2))

        # Step 2: Initial Horizontal Velocity
        initial_velocity_x = self.target_distance / time_of_flight

        # Estimate the effect of air resistance on horizontal velocity
        air_resistance_effect_x = AIR_RESISTANCE_COEFF * initial_velocity_x
        initial_velocity_x += air_resistance_effect_x  # Adjust for air resistance

        # Step 3: Initial Vertical Velocity
        # For vertical motion, use: velocity = gravity * time
        # Since time to peak is half of time_of_flight
        initial_velocity_y = GRAVITY * (time_of_flight / 2)

        # Estimate the effect of air resistance on vertical velocity
        air_resistance_effect_y = AIR_RESISTANCE_COEFF * initial_velocity_y
        initial_velocity_y += air_resistance_effect_y  # Adjust for air resistance

        # Step 4: Calculate Angle and Velocity
        angle = math.degrees(math.atan2(initial_velocity_y, initial_velocity_x))
        velocity = math.sqrt(initial_velocity_x**2 + initial_velocity_y**2)
        
        return velocity, angle

    def apply_gravity(self):
        gravitational_acceleration = GRAVITY
        self.acceleration_y -= gravitational_acceleration

    def apply_air_resistance(self):
        # Calculate the acceleration due to air resistance
        air_resist_accel_x = AIR_RESISTANCE_COEFF * self.velocity_x
        air_resist_accel_y = AIR_RESISTANCE_COEFF * self.velocity_y

        # Apply the acceleration in the opposite direction of motion
        self.acceleration_x -= air_resist_accel_x
        self.acceleration_y -= air_resist_accel_y

    def apply_propulsion(self):
        if self.has_fuel:
            acceleration_due_to_propulsion = self.propulsion_force / self.weight
            self.acceleration_x += acceleration_due_to_propulsion * math.cos(math.radians(self.angle))
            self.acceleration_y += acceleration_due_to_propulsion * math.sin(math.radians(self.angle))
            # Optionally decrease fuel here
            self.has_fuel = False

    def update(self):
        # Update the acceleration
        self.apply_gravity()
        self.apply_air_resistance()
        self.apply_propulsion()

        # Update the velocity based on acceleration
        self.velocity_x += self.acceleration_x * TIME_PER_FRAME
        self.velocity_y += self.acceleration_y * TIME_PER_FRAME

        # Update the position based on velocity
        self.x += self.velocity_x * TIME_PER_FRAME
        self.y -= self.velocity_y * TIME_PER_FRAME  # Negative because Pygame's Y-axis is inverted

        # Update angle based on velocity
        self.angle = math.degrees(math.atan2(self.velocity_y, self.velocity_x))

        # Reset accelerations for the next frame
        self.acceleration_x = 0
        self.acceleration_y = 0

    def draw(self, win):
        pygame.draw.circle(win, RED, (int(self.x), int(self.y)), 5)

class Target:
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = HEIGHT  # Place the target on the ground
        self.size = size
        self.hit = False

    def draw(self, win):
        if not self.hit:
            pygame.draw.rect(win, GREEN, (self.x - self.size//2, self.y - self.size//2, self.size, self.size))

def check_collision(missile, target):
    return not target.hit and abs(missile.x - target.x) < target.size and abs(missile.y - target.y) < target.size

def main():
    run = True
    clock = pygame.time.Clock()
    missile = Missile(propulsion_force=100, weight=10, target_distance=100)
    target = Target(x=100, y=HEIGHT)  # Place target at the missile's target distance

    while run:
        clock.tick(FPS)

        WIN.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        missile.update()
        if check_collision(missile, target):
            target.hit = True

        missile.draw(WIN)
        target.draw(WIN)

        if missile.x > WIDTH or missile.y > HEIGHT or missile.y < 0:
            missile = Missile(propulsion_force=100, weight=10, target_distance=500)
            target = Target(x=500, y=HEIGHT)  # Reset target

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()

"""
Enhancements: Refine the simulation by adding more features or improving the visuals.
Optimization: Calculate the cost of the missile based on the acceleration, weight, and distance.

# Print statements for debugging
print(f"Position: ({self.x}, {self.y})")
print(f"Velocity: ({self.velocity_x}, {self.velocity_y})")
print(f"Acceleration: ({self.acceleration_x}, {self.acceleration_y})")


Set a fuel object and engine object for the missile. The engine should use the fuel at launch for a set rate to increase the acceleration until the set amount of fuel is used up.

ballistic missile vs cruise missile

Assumptions about Trajectory: The code seems to assume that the missile will follow a parabolic trajectory. This might not hold true in all cases, especially if the missile is self-propelled and can adjust its trajectory mid-flight.
Propulsion Force: If the missile is self-propelled, add a force that continuously accelerates the missile until the fuel is depleted or until a certain point in its trajectory.

aerodynamic lift
inertia
rotation
dynamics of flight control systems
orientation and using control surfaces to steer
thrust vectoring
control surfaces
interactions with the weather and environment

Realistic Initial Conditions: Revise the calculation of initial velocity and angle. Often, these are determined through optimization techniques considering the desired range, maximum height, and other trajectory characteristics.
Configurability: Allow the user to adjust parameters like missile weight, propulsion force, drag coefficient, etc., to see how they affect the trajectory.

Please solve differential equations to get the time of flight.
# Simplified formula to estimate launch angle (ignores air resistance)
angle = math.degrees(math.atan((v**2 - math.sqrt(v**4 - g * (g * d**2 + 2 * HEIGHT * v**2))) / (g * d)))

A GuidanceSystem class with methods for updating missile heading and position. This modularity allows for different guidance strategies to be implemented and tested with the CruiseMissile class.

Control when to use the fuel and in what direction
Max acceleration and max velocity
Anti-ballistic missile

https://www.mwrf.com/markets/defense/article/21848658/the-3-major-phases-of-effective-missile-defense-systems

https://zh.wikipedia.org/zh-tw/%E6%98%9F%E7%90%83%E5%A4%A7%E6%88%98%E8%AE%A1%E5%88%92

https://zh.wikipedia.org/zh-tw/%E6%B4%B2%E9%9A%9B%E5%BD%88%E9%81%93%E9%A3%9B%E5%BD%88

https://en.wikipedia.org/wiki/Ballistic_missile_flight_phases

https://www.atlantis-press.com/article/25868869.pdf
"""

"""
Proportional navigation is a guidance law used in missile technology to ensure that a missile intercepts a moving target. It's a concept widely used in both surface-to-air and air-to-air missiles, among other types. The underlying principle of proportional navigation (PN) is not to steer the missile directly at the target, but rather to maneuver it based on the target's motion to maintain a collision course.

Here's how it works in simplified terms:

1. **Line of Sight (LOS) Rate**: Proportional navigation focuses on the Line of Sight (LOS) angle between the missile and the target. The LOS is the straight line that connects the missile and the target. The guidance system continually measures the rate at which this angle is changing.

2. **Proportional Response**: The missile is commanded to turn at a rate that is proportional to the LOS rate of change. The constant of proportionality is called the Navigation Constant or Navigation Ratio (usually denoted by "N"). A common value for N is 3, which means the missile will turn at a rate three times that of the LOS rate of change. This factor ensures that the missile leads the target, anticipating where it will be in the future rather than where it is currently.

3. **Collision Course**: The result of this type of guidance law is that the missile will fly a path that brings it to an intersection with the target's path, ideally at the same time the target arrives at that point. This is because the guidance logic mathematically drives the missile's velocity vector to align with the LOS to the target as they converge.

4. **Advantages**: Proportional navigation is robust and simple to implement. It does not require the missile to know the target's velocity or predict its future position, only the change in the LOS angle. PN is also less sensitive to measurement errors and uncertainties about the target's motion compared to other guidance methods.

5. **Energy Management**: While PN is about the direction of flight, a missile also has to manage its energy (speed and altitude) to ensure it has enough kinetic energy to reach the target. This aspect is handled by other systems on the missile, which work in conjunction with the PN guidance.

In essence, proportional navigation is an elegant solution to the interception problem because it automatically compensates for changes in the target's movement without requiring complex predictive calculations. It leverages the geometry of the situation to guide the missile to the right interception point, making it a very effective and widely used missile guidance method.
"""

"""
Ballistic missiles and cruise missiles are two distinct types of weapons systems used for delivering payloads over long distances, but they have different flight profiles, propulsion methods, and guidance systems.

**Ballistic Missile:**

1. **Flight Trajectory:** A ballistic missile follows a ballistic trajectory to reach its target. This means that it is launched directly into the upper layers of the Earth's atmosphere or beyond (into space) and travels outside the atmosphere before re-entering and descending onto its target. Its path is largely determined by gravity and is therefore predictable once the missile is in flight.

2. **Propulsion:** Ballistic missiles are powered by rocket engines and boosters that provide the necessary thrust to escape the Earth's gravity. They only burn their fuel for a short period during the launch phase; after that, they follow an unpowered trajectory.

3. **Guidance:** The guidance system of a ballistic missile is mainly used during the initial powered phase of flight. Once it exits the atmosphere, it follows a predetermined path that cannot be altered significantly, making it less flexible in targeting after launch.

**Cruise Missile:**

1. **Flight Trajectory:** A cruise missile is designed to fly at a constant altitude, usually quite low to the ground, following the terrain to avoid detection. Its flight path can be programmed to be highly erratic to evade enemy defenses.

2. **Propulsion:** Cruise missiles are powered throughout their flight by jet engines, much like an airplane. They can be launched from land, sea, or air platforms.

3. **Guidance:** They have advanced guidance systems that can receive updates during flight. They often use a combination of GPS, inertial navigation, terrain contour matching (TERCOM), and sometimes image matching systems (DSMAC) to find their target.

Ballistic Missile

**Boost Phase:**
The boost phase for a ballistic missile is the initial stage of its flight. It starts from the moment of launch and continues until the rocket engines stop firing and the missile ceases to gain altitude. During this phase, the missile is powered by its rocket boosters, which provide the necessary thrust to leave the Earth’s atmosphere. The missile's trajectory during the boost phase is somewhat adjustable, allowing for corrections and stabilization as it heads towards the edge of space. This is also the most vulnerable phase of the missile's flight, as it is relatively slow and can be tracked and potentially intercepted.

**Midcourse Phase:**
After the boost phase, the missile enters the midcourse phase, where it follows a ballistic trajectory—essentially coasting through space under the influence of gravity. This phase can last anywhere from a few minutes to over half an hour, depending on the missile's range. During this phase, the missile is outside of the Earth's atmosphere and it's in a free-flight trajectory. Some ballistic missiles may deploy decoys during the midcourse phase to confuse missile defense systems. The guidance system makes minor adjustments using small thrusters, if necessary, to correct the flight path for any initial launch errors or to adjust for conditions in space.

**Terminal Phase:**
The terminal phase begins once the missile re-enters the Earth's atmosphere and heads towards its target. It is characterized by re-entry at extremely high speeds, with the warhead (or multiple warheads in the case of MIRV—Multiple Independently targetable Reentry Vehicles) encountering atmospheric resistance and heating. This phase is relatively short, but it is also the most challenging for missile defenses to intercept the incoming warhead(s) due to the high velocity and sometimes the presence of countermeasures. The warheads are generally not powered during this phase and rely on their speed and ballistic trajectory to reach the target. The precision of the strike depends on the accuracy of the missile's guidance system during the launch and any terminal guidance that might be employed, such as maneuverable reentry vehicles (MaRVs).

In contrast to a cruise missile, a ballistic missile's trajectory is much less flexible once it's in its midcourse phase, making its path predictable if not altered by countermeasures or sophisticated guidance systems like MaRVs. Ballistic missiles are also typically much faster in their terminal phase compared to cruise missiles, which makes them more challenging to defend against.

Cruise missile

Continuous propulsion, lower altitude flight path, and advanced guidance systems.

Launch Phase:
Think of the launch phase as the missile's kickoff moment. If it's a ground or sea-launched missile, it typically ignites a solid rocket booster that propels it to the required altitude and velocity. This phase is all about getting the missile up to speed and at the right height to start its main journey.

Cruise Phase:
During the cruise phase, the missile settles into its main travel mode. The missile follows pre-set navigational waypoints. It maintains a low altitude, often employing terrain-hugging techniques to to avoid detection. The missile's sophisticated guidance systems, which can include inertial navigation coupled with terrain contour matching (TERCOM), satellite guidance (like GPS) and other sensors, constantly monitor and correct its flight path, ensuring it stays on course.

Terminal Phase:
In the terminal phase of a cruise missile's flight, the focus is on pinpoint accuracy and evasion. The missile may engage additional guidance systems, like radar or infrared homing, to precisely target its destination. It maneuvers to avoid defenses, possibly using decoys and electronic warfare to outwit enemy systems. Some missiles perform a "pop-up" maneuver to evade defenses before making a steep dive towards the target. This phase culminates with the missile making fine adjustments to ensure it hits the target precisely, coordinating all systems for a successful strike.
"""