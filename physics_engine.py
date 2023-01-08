"""
major pillars of an engine: input, graphics, physics, entities, and audio
"""

"""
To simulate Newtonian physics for particles using Python, some of the key components of your physics engine might include:

A class or data structure to represent a particle, which will store information such as the particle's position, velocity, mass, and any other properties that are relevant to your simulation.

Functions to calculate the forces acting on a particle, such as gravity, friction, and spring forces. These forces will depend on the specific details of your simulation, such as the mass and acceleration of the particles, the coefficients of friction and restitution, and any other relevant parameters.

Functions to update the position and velocity of a particle based on the forces acting on it. This might involve using a numerical integration method such as the Euler method or the Runge-Kutta method to solve the equations of motion for the particle.

Code to handle user input and display the results of the simulation on the screen. This might involve creating a GUI using a library such as PyQt or PyGTK, or using a library such as Pygame to create a more interactive simulation.

Test cases and other debugging tools to ensure that your physics engine is functioning correctly. This might involve writing test cases to verify that your code is producing the expected results, as well as using tools such as a debugger or print statements to help you troubleshoot any issues that arise.
"""

"""
It is often useful to divide the code for a physics engine into separate modules or classes for handling dynamics, collision detection, and collision response. Here's a brief overview of what each of these components might do:

Dynamics: This component is responsible for updating the position and velocity of the simulated objects based on the forces acting on them. It might use a numerical integration method such as the Euler method or the Runge-Kutta method to solve the equations of motion for each object.

Collision detection: This component is responsible for identifying when two or more objects have collided. It might use geometric techniques such as bounding box intersection or ray casting to detect collisions, or it might use more advanced techniques such as spatial partitioning or shape-based collision detection.

Collision response: This component is responsible for simulating the effects of a collision on the objects involved. It might update the objects' velocities and positions based on the coefficients of restitution and friction, or it might apply additional forces such as impulses or torques to the objects.
"""

"""
Users can configure it at several different levels, such as whether they want a collider or want to respond to collisions or want to simulate dynamics or whether they want dynamics, but not gravity or they want to include rotation, multiple contact point collisions, or constrained simulation.
"""


import numpy as np

# dynamics
import math

class Particle:
    def __init__(self, pos, vel, acc, mass):
        self.pos = pos  # 3D position vector
        self.vel = vel  # 3D velocity vector
        self.acc = acc  # 3D acceleration vector
        self.mass = mass  # Scalar mass
        
    def update(self, dt):
        """Update the position and velocity of the particle using the Euler method"""
        # Update the velocity of the particle
        self.vel = self.vel + self.acc * dt
        
        # Update the position of the particle
        self.pos = self.pos + self.vel * dt

def calculate_forces(particles):
    forces = []
    for particle in particles:
        force = np.zeros(3)  # 3D force vector

        # Add any other forces here
        force += np.array([0, -9.81, 0]) * particle.mass  # Gravity
        force += np.array([0, 0, 0])  # Other forces

        forces.append(force)
    return forces

def update_particles(particles, dt, forces):
    for i in range(len(particles)):
        # Update the acceleration of the particle based on the forces acting on it
        particles[i].acc = forces[i] / particles[i].mass

        # Update the position and velocity of the particle using the Euler method
        particles[i].update(dt)

def simulate(particles, dt):
    # Calculate the forces acting on each particle
    forces = calculate_forces(particles)
    # Update the position, velocity and acceleration of each particle
    update_particles(particles, dt, forces)
    
def display(particles, t):
    """Display the current state of the simulation"""
    print("Time: {}".format(t))
    for particle in particles:
        print("Position: {}".format(particle.pos))
        print("Velocity: {}".format(particle.vel))
        print("Acceleration: {}".format(particle.acc))
        print("Mass: {}".format(particle.mass))
        print("")
    print("")

def main():
    
    # Set up the simulation
    dt = 0.01  # Time step
    t = 0  # Current time
    t_max = 10  # Maximum time
    
    # Create a list of particles
    particles = [Particle(np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 0, 0]), 1),
                Particle(np.array([1, 1, 1]), np.array([0, 0, 0]), np.array([0, 0, 0]), 2)]

    # Simulate the particles for a fixed amount of time
    while t < t_max:
        simulate(particles, dt)
        # Display the current state of the simulation (e.g. using a GUI or console output)
        display(particles, t)
        # Update the current time
        t += dt

if __name__ == "__main__":
    main()










import math

class Circle:
    def __init__(self, x, y, radius, mass, velocity_x, velocity_y, elasticity):
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.elasticity = elasticity
        
    def handle_collision(self, other_shape):
        # Trigger an animation
        # animation.play()
        # Apply damage to the other shape
        other_shape.health -= 10

class Rectangle:
    def __init__(self, x, y, width, height, mass, velocity_x, velocity_y, elasticity):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mass = mass
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.elasticity = elasticity
        
    def handle_collision(self, other_shape):
        # Handle any additional effects of the collision (e.g., playing a sound, triggering an animation, applying damage)
        pass

def detect_collision(shape1, shape2):
    if isinstance(shape1, Circle) and isinstance(shape2, Circle):
        # Calculate the distance between the two centers
        dx = shape1.x - shape2.x
        dy = shape1.y - shape2.y
        distance = math.sqrt(dx**2 + dy**2)

        # Check if the distance is smaller than the sum of the radii
        if distance < shape1.radius + shape2.radius:
            return True
        else:
            return False

    elif isinstance(shape1, Rectangle) and isinstance(shape2, Rectangle):
        # Check if the rectangles overlap horizontally
        if abs(shape1.x - shape2.x) < shape1.width + shape2.width:
            # Check if the rectangles overlap vertically
            if abs(shape1.y - shape2.y) < shape1.height + shape2.height:
                return True
            else:
                return False
        else:
            return False

    elif isinstance(shape1, Circle) and isinstance(shape2, Rectangle):
        # Check if the circle is inside the rectangle
        if (shape1.x + shape1.radius >= shape2.x and
            shape1.x - shape1.radius <= shape2.x + shape2.width and
            shape1.y + shape1.radius >= shape2.y and
            shape1.y - shape1.radius <= shape2.y + shape2.height):
            return True
        else:
            # Check if the circle intersects any of the rectangle's sides
            sides = [(shape2.x, shape2.y, shape2.x + shape2.width, shape2.y),
                     (shape2.x + shape2.width, shape2.y, shape2.x + shape2.width, shape2.y + shape2.height),
                     (shape2.x + shape2.width, shape2.y + shape2.height, shape2.x, shape2.y + shape2.height),
                     (shape2.x, shape2.y + shape2.height, shape2.x, shape2.y)]
            for x1, y1, x2, y2 in sides:
                if line_intersect_circle(x1, y1, x2, y2, shape1.x, shape1.y, shape1.radius):
                    return True
            return False

    elif isinstance(shape1, Rectangle) and isinstance(shape2, Circle):
        return detect_collision(shape2, shape1)

def line_intersect_circle(x1, y1, x2, y2, cx, cy, radius):
    # Check if the line is horizontal
    if y1 == y2:
        # Check if the circle is above or below the line
        if (cy < y1 and cy + radius < y1) or (cy > y1 and cy - radius > y1):
            return False
        else:
            # Check if the circle intersects the line
            if (cx < x1 and cx + radius < x1) or (cx > x2 and cx - radius > x2):
                return False
            else:
                return True
    # Check if the line is vertical
    elif x1 == x2:
        # Check if the circle is to the left or right of the line
        if (cx < x1 and cx + radius < x1) or (cx > x1 and cx - radius > x1):
            return False
        else:
            # Check if the circle intersects the line
            if (cy < y1 and cy + radius < y1) or (cy > y2 and cy - radius > y2):
                return False
            else:
                return True
    else:
        # Calculate the slope and y-intercept of the line
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Calculate the distance from the center of the circle to the line
        distance = abs(cy - m * cx - b) / math.sqrt(m**2 + 1)

        # Check if the distance is smaller than the radius
        if distance < radius:
            return True
        else:
            return False


def handle_collision(shape1, shape2):
    # Determine the collision normal
    normal = (0, 0)
    # Initialise the distance to 0
    distance = 0
    if isinstance(shape1, Circle) and isinstance(shape2, Circle):
        # Calculate the collision normal as the unit vector pointing from the center of shape1 to the center of shape2
        dx = shape2.x - shape1.x
        dy = shape2.y - shape1.y
        distance = math.sqrt(dx**2 + dy**2)
        normal = (dx / distance, dy / distance)
    elif isinstance(shape1, Rectangle) and isinstance(shape2, Rectangle):
        # Calculate the collision normal as the unit vector pointing from the center of shape1 to the center of shape2
        dx = shape2.x - shape1.x
        dy = shape2.y - shape1.y
        distance = math.sqrt(dx**2 + dy**2)
        normal = (dx / distance, dy / distance)
    elif isinstance(shape1, Circle) and isinstance(shape2, Rectangle):
        # Calculate the collision normal as the unit vector pointing from the center of the circle to the nearest point on the rectangle
        closest_x = max(shape2.x, min(shape1.x, shape2.x + shape2.width))
        closest_y = max(shape2.y, min(shape1.y, shape2.y + shape2.height))
        dx = closest_x - shape1.x
        dy = closest_y - shape1.y
        distance = math.sqrt(dx**2 + dy**2)
        normal = (dx / distance, dy / distance)
    elif isinstance(shape1, Rectangle) and isinstance(shape2, Circle):
        # Calculate the collision normal as the unit vector pointing from the center of the circle to the nearest point on the rectangle
        closest_x = max(shape1.x, min(shape2.x, shape1.x + shape1.width))
        closest_y = max(shape1.y, min(shape2.y, shape1.y + shape1.height))
        dx = closest_x - shape2.x
        dy = closest_y - shape2.y
        distance = math.sqrt(dx**2 + dy**2)
        normal = (dx / distance, dy / distance)
    
    # Calculate the new velocity of each object based on the collision normal, their masses, and their elasticity
    mass1 = shape1.mass
    mass2 = shape2.mass
    elasticity1 = shape1.elasticity
    elasticity2 = shape2.elasticity
    v1x, v1y = shape1.velocity_x
    v2x, v2y = shape2.velocity_y
    v1x, v1y, v2x, v2y = calculate_new_velocity(mass1, elasticity1, v1x, v1y, mass2, elasticity2, v2x, v2y, normal)
    shape1.velocity_x, shape1.velocity_y = (v1x, v1y)
    shape2.velocity_x, shape2.velocity_y = (v2x, v2y)

    # Update the position of each object to prevent them from intersecting
    overlap = 0
    if isinstance(shape1, Circle) and isinstance(shape2, Circle):
        overlap = shape1.radius + shape2.radius - distance
    elif isinstance(shape1, Rectangle) and isinstance(shape2, Rectangle):
        overlap = (shape1.width + shape2.width - abs(shape1.x - shape2.x)) / 2 # or (shape1.width / 2) + (shape2.width / 2)
    elif isinstance(shape1, Circle) and isinstance(shape2, Rectangle):
        overlap = shape1.radius + (shape2.width / 2) - distance # width vs height
    elif isinstance(shape1, Rectangle) and isinstance(shape2, Circle):
        overlap = (shape1.width / 2) + shape2.radius - distance
    shape1.x -= normal[0] * overlap
    shape1.y -= normal[1] * overlap
    shape2.x += normal[0] * overlap
    shape2.y += normal[1] * overlap

    # Handle any additional effects of the collision (e.g., playing a sound, triggering an animation, applying damage)
    shape1.handle_collision(shape2)
    shape2.handle_collision(shape1)

def calculate_new_velocity(mass1, elasticity1, v1x, v1y, mass2, elasticity2, v2x, v2y, normal):
    # Calculate the velocity of the center of mass
    v_cm_x = (mass1 * v1x + mass2 * v2x) / (mass1 + mass2)
    v_cm_y = (mass1 * v1y + mass2 * v2y) / (mass1 + mass2)

    # Calculate the relative velocity of the objects
    v_rel_x = v1x - v2x
    v_rel_y = v1y - v2y

    # Calculate the component of the relative velocity along the collision normal
    v_rel_normal = v_rel_x * normal[0] + v_rel_y * normal[1]

    # Calculate the new relative velocity
    e = elasticity1 * elasticity2
    v_rel_x_new = v_rel_x - v_rel_normal * normal[0] * (1 + e)
    v_rel_y_new = v_rel_y - v_rel_normal * normal[1] * (1 + e)
    
    # Calculate the new velocity of each object
    v1x_new = v_cm_x + v_rel_x_new
    v1y_new = v_cm_y + v_rel_y_new
    v2x_new = v_cm_x - v_rel_x_new
    v2y_new = v_cm_y - v_rel_y_new

    return v1x_new, v1y_new, v2x_new, v2y_new


# Define the shapes
shape1 = Circle(0, 0, 10, 0, 0, 1, 1)
shape2 = Circle(30, 0, 10, 0, 0, 1, 1)

# Check for a collision and handle it if one is detected
if detect_collision(shape1, shape2):
    handle_collision(shape1, shape2)

# Define the shapes
shape1 = Rectangle(0, 0, 20, 20, 0, 0, 1, 1)
shape2 = Rectangle(30, 0, 20, 20, 0, 0, 1, 1)

# Check for a collision and handle it if one is detected
if detect_collision(shape1, shape2):
    handle_collision(shape1, shape2)

