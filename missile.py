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
        if self.has_fuel and self.x < self.target_distance:
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
        self.velocity_x += self.acceleration_x * dt
        self.velocity_y += self.acceleration_y * dt

        # Update the position based on velocity
        self.x += self.velocity_x * dt
        self.y -= self.velocity_y * dt  # Negative because Pygame's Y-axis is inverted

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
    missile = Missile(propulsion_force=100, weight=10, target_distance=300)
    target = Target(x=300, y=HEIGHT)  # Place target at the missile's target distance

    while run:
        global dt
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

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