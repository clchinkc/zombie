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

"""
Encapsulation: Each class should have a clear responsibility and should not expose its internal state or implementation details. Avoid using public class attributes and instead use properties or getter/setter methods to access an object's state.
Single Responsibility Principle: Each class should have a single responsibility, and that responsibility should be entirely encapsulated by the class.
Dependency Injection: Instead of creating dependencies inside the class, pass them in as constructor arguments. This makes it easier to test the class and to swap out one implementation for another.
Separation of Concerns: Keep physics-related functionality in the physics-related classes, and collision-related functionality in the collision-related classes. This will make it easier to understand and maintain the code, and also allows for easier modification of specific parts of the engine without affecting the rest.
Composition over Inheritance: Instead of inheriting the functionality of other classes, use composition to include the functionality of other classes. This allows for more flexibility and can make the code more reusable.
Use Interfaces: Create interfaces for different types of objects in the physics engine, such as colliders and solvers. This allows for more flexibility and makes it easier to add new types of objects without modifying existing code.
Use of Design Patterns: Use design patterns such as the State pattern, Strategy pattern, or Observer pattern to make the code more flexible and maintainable.
Use of Data Structures: Proper usage of data structures like spatial partitioning, broad-phase collision detection, and narrow-phase collision detection can optimize collision detection and response.
Code documentation: Document the code with comments and clear naming convention to make it easy for others to understand and to maintain the code.
Use of testing: Use testing frameworks to test the different components of the engine, this will help to make sure that the engine is working as expected and to catch any bugs early on.
"""

"""
A modern physics engine typically has the following classes and methods:
Vector3: Represents a 3D vector.
Methods:
__init__(self, x, y, z): Initializes the vector with the given x, y, and z values.
length(self): Returns the length of the vector.
normalize(self): Returns a new vector that is a normalized version of the original vector.
dot(self, other): Returns the dot product of the vector with another vector.
cross(self, other): Returns the cross product of the vector with another vector.
__add__(self, other): Overloads the + operator to add two vectors.
__sub__(self, other): Overloads the - operator to subtract two vectors.
__mul__(self, scalar): Overloads the * operator to scale a vector by a scalar.
Quaternion: Represents a quaternion.
Methods:
__init__(self, x, y, z, w): Initializes the quaternion with the given x, y, z, and w values.
length(self): Returns the length of the quaternion.
`normalize(self): Returns a new quaternion that is a normalized version of the original quaternion. - inverse(self): Returns the inverse of the quaternion. - multiply(self, other): Returns the product of the quaternion with another quaternion. - rotate_vector(self, vector): Rotates a vector by the quaternion. - slerp(self, other, t)`: Returns the spherical linear interpolation between two quaternions.

Transform: Represents the location and orientation of an object in 3D space.
Methods:
__init__(self, position, scale, rotation): Initializes the transform with the given position, scale, and rotation.
get_local_to_world_matrix(self): Returns the transformation matrix that transforms the object from its local space to world space.
get_world_to_local_matrix(self): Returns the transformation matrix that transforms the object from world space to its local space.
Collider: Represents the collision detection component of an object.
Methods:
__init__(self): Initializes the collider.
test_collision(self, other, transform, other_transform): Tests for a collision between the collider and another collider.
get_bounding_volume(self, transform): Returns the bounding volume of the collider in world space.
RigidBody: Represents the dynamics component of an object.
Methods:
__init__(self, mass, collider, transform): Initializes the rigid body with the given mass, collider, and transform.
apply_force(self, force): Applies a force to the rigid body.
apply_impulse(self, impulse): Applies an impulse to the rigid body.
update(self, dt): Updates the position and velocity of the rigid body based on forces and impulses.
PhysicsWorld: Represents the physics simulation of the game world.
Methods:
__init__(self): Initializes the physics world.
add_rigid_body(self, body):Adds a rigid body to the physics world.
- remove_rigid_body(self, body): Removes a rigid body from the physics world.
- step(self, dt): Steps the simulation forward by the given time step.
- set_gravity(self, gravity): Sets the gravity vector for the physics world.
- get_gravity(self): Returns the current gravity vector of the physics world.
- get_rigid_bodies(self): Returns a list of all the rigid bodies in the physics world.

CollisionWorld: Represents the collision detection and response component of the game world.
Methods:
__init__(self): Initializes the collision world.
add_collision_object(self, obj): Adds an object to the collision world.
remove_collision_object(self, obj): Removes an object from the collision world.
set_collision_callback(self, callback): Sets the callback function to be called when a collision occurs.
detect_collisions(self): Detects collisions between objects in the collision world.
solve_collisions(self): Solves collisions between objects in the collision world.
get_collision_objects(self): Returns a list of all the objects in the collision world.
Solver: Represents the collision response algorithm used in the game world.
Methods:
__init__(self): Initializes the solver.
solve(self, collisions, dt): Solves the given collisions using the algorithm implemented in the solver.
"""
"""
A physics engine for a grid-based game will typically have the following components:

Collision Detection: This component is responsible for determining when two objects in the game world have collided. This can be done using various methods, such as bounding box collision detection, where the game checks if the rectangular bounding boxes of two objects overlap.
Collider class: This class represents a collider component that can be attached to a game object. It contains information about the shape and size of the collider, as well as methods for performing collision detection.
CollisionDetector class: This class contains methods for performing collision detection between different colliders. It can check for collisions between two colliders by comparing their positions and shapes.
checkCollision(collider1: Collider, collider2: Collider) method: This method is responsible for checking if two colliders are colliding. It takes two colliders as input and returns a Boolean value indicating whether a collision has occurred.
onCollisionEnter(collider1: Collider, collider2: Collider) and onCollisionExit(collider1: Collider, collider2: Collider) methods: These methods are called when a collision is first detected and when two colliders stop colliding. They can be used to trigger events or apply effects when a collision occurs.
getCollisionNormal(collider1: Collider, collider2: Collider): Method which returns the normal of the collision, which can be used for resolving the collision response.
Note that this is a general outline and the actual implementation will depend on the design and the language of the engine. Some physics engine might use other data structures like Quadtree, Spatial Hashmap etc to handle large number of colliders.

Grid-based collision response: This component is responsible for resolving collisions that occur between objects in the game world. This can include pushing objects out of each other, or making one object pass through the other. Because the game world is divided into a grid, the collision response may need to take the grid structure into account.
Grid class: This class represents the grid-based game world and contains information about the size and layout of the grid. It can be used to map between world coordinates and grid coordinates.
GridCell class: This class represents a single cell in the grid and contains information about the terrain or objects that occupy that cell.
resolveCollision(collider1: Collider, collider2: Collider) method: This method is responsible for resolving collisions that occur between objects in the game world. It takes two colliders as input and modifies their positions and velocities to resolve the collision.
getValidMovement(collider: Collider, movement: Vector) method: This method takes a collider and a movement vector as input and returns a modified movement vector that takes into account any collisions that would occur if the collider were to move in that direction.
moveCollider(collider: Collider, movement: Vector) method: This method is used to move a collider in the game world, taking into account any collisions that might occur. It can be used to update the position of a collider based on its velocity and the time step of the simulation.
Note that this is a general outline and the actual implementation will depend on the design and the language of the engine. Some engine might use other methodologies like sweep and prune, spatial partitioning etc to handle collision response.

Rigid Body Dynamics: This component is responsible for simulating the behavior of objects in the game world when they are subject to various forces such as gravity, friction, and wind resistance. This can include calculating the motion of objects in the game world and how they react to different forces.
DynamicsWorld class: This class represents the world in which all the dynamic objects exist. It contains all the objects and simulated the dynamics of the objects under the effect of forces, torques and constraints.
RigidBody class: This class represents a rigid body in the game world and contains information about its mass, velocity, position, and rotation. It also contains methods for applying forces and torques to the rigid body.
Constraint classes: This classes handle different types of constraints like walls, hinge, ball-socket, slider etc that can be added to the dynamic objects. These constraints can be used to simulate things like joints between objects, motors, and brakes.
Solver classes: These classes handle the numerical integration of the dynamics equations. Different types of solvers can be used for different types of dynamics simulations, such as Verlet integration or Runge-Kutta methods.
addObject(object: RigidBody) and removeObject(object: RigidBody): These methods are used to add or remove dynamic objects from the world.
applyForce(object: RigidBody, force: Vector), applyTorque(object: RigidBody, torque: Vector) and applyImpulse(object: RigidBody, impulse: Vector): These methods are used to apply different types of forces, torque and impulse from the outside to the objects for collision detection.
integrate(rigidBody: RigidBody, dt: float) method: This method is used to update the position and velocity of a rigid body based on the forces and torques acting on it. It takes the rigid body and the time step of the simulation as input and modifies the rigid body's position, velocity and rotation accordingly, to handle collision response.
Note that this is a general outline and the actual implementation will depend on the design and the language of the engine. Some engines might use other methodologies like Verlet Integration, Runge-Kutta etc to handle rigid body dynamics.
Note that this is a general outline and the actual implementation will depend on the design and the language of the engine. Some engine might use other methodologies like constraint based dynamics, hybrid dynamics etc to handle dynamics simulation.


Grid-based Simulation: The physics engine will also include some specific simulation that caters to the grid based structure. For instance, using grid-based spatial partitioning to handle large number of objects and their collision detection.
GridSimulation class: This class is responsible for updating the state of the game world based on the grid structure. It can be used to simulate things like fluid flow, sand falling, and other grid-based phenomena.
update(grid: Grid, dt: float) method: This method updates the state of the grid for a given time step. It can be used to simulate things like fluid flow, sand falling, and other grid-based phenomena.
applyConstraints(grid: Grid) method: This method applies constraints to the simulation to maintain the integrity of the grid structure. This can include things like maintaining the correct pressure, preventing cells from becoming empty or overfull, and ensuring that cells remain connected.
spread(grid: Grid, material: Material, position: Vector) method: This method spreads a material (e.g water, sand, oil) over the grid starting from a given position.
getNeighborCells(grid: Grid, position: Vector) method: This method returns the neighboring cells of a given position, taking into account the grid structure. This can be used to implement things like fluid flow or the spread of materials.
Note that this is a general outline and the actual implementation will depend on the design and the language of the engine. Some engines might use other data structures like Quadtree, Spatial Hashmap etc to handle grid-based simulation.

Optimization: The engine will also have some optimization techniques to make the physics simulation more efficient. This can include techniques such as multithreading, spatial partitioning, and caching.
PerformanceMonitor class: This class can be used to track and measure the performance of the physics engine, such as the frame rate, CPU usage, and memory usage.
Profiler class: This class can be used to profile different parts of the physics engine to identify performance bottlenecks. It can measure the time taken by different functions and methods and output this information for analysis.
Multithreading class: This class can be used to divide the workload of the physics engine across multiple threads, which can improve performance on multi-core processors.
SpatialPartitioning class: This class can be used to organize the colliders in the game world into spatial partitions, such as grids or bounding volume hierarchies, to improve performance when checking for collisions.
Caching class: This class can be used to cache the results of expensive calculations, such as collision detection and rigid body dynamics, so that they can be reused without having to be recalculated.

Overall, the physics engine will be adapted to work in the grid-based environment, while still being able to provide realistic and interactive physical simulation.
"""
"""
How can I use this physics engine in a grid-based game?
Create a Grid class that represents the game's grid. This class should have methods for initializing the grid, accessing the grid cells, and determining the grid cell that a particular point is in.
Use the Transform class to store the position of each game object in the grid. When an object moves, update its position in the Transform class, and then use the Grid class to determine which grid cell the object is now in.
Use the RigidBody class to simulate the dynamics of the objects in the game. Apply forces and impulses to the objects as needed to move them around the grid.
Use the CollisionWorld class to detect and resolve collisions between objects in the game. This class should be able to detect collisions between objects in different grid cells and should be able to resolve collisions by moving the objects apart.
Use the PhysicsWorld class to simulate the game physics and handle collision detection. The PhysicsWorld class will update the position and velocity of objects based on the forces acting on them and will detect and resolve collisions between objects.
Create a Game class that will handle the game loop and the game logic. This class should use the PhysicsWorld class to update the game state, and the Grid class to check for and handle collisions.
Finally, create a main loop that calls the update method of the Game class, and render the game using the Grid and Transform classes.
You can use spatial partitioning data structures like Quadtree, Octree, Grid or BSP to optimize collision detection and response.
Use the grid-based movement mechanics to constrain the movement of objects, you can also use grid-based collision detection to improve performance.
You can also use the grid-based approach to implement the pathfinding algorithm in the game.
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

def dynamic():
    
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


# dynamic()









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

def collision():
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








# from cpp



# dynamics
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# dynamics
class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

# Collision detection
class Transform: # describes an objects location and orientation in 3D space
    def __init__(self, Position, Scale, Rotation):
        self.Position = Position # Vector3
        self.Scale = Scale # Vector3
        self.Rotation = Rotation # Quaternion

class Object:    
    def __init__(self, position, velocity, force, mass, Collider, Transform):
        self.Position = position
        self.Velocity = velocity
        self.Force = force
        self.Mass = mass
        self.Collider = Collider # Collider
        self.Transform = Transform # Transform

# collision detection
class CollisionPoints:
    def __init__(self, A, B, Normal, Depth, HasCollision):
        self.A = A # furthest point of A into B # Vector3
        self.B = B # furthest point of B into A # Vector3
        self.Normal = Normal # B – A normalized # Vector3
        self.Depth = Depth # length of B – A # float
        self.HasCollision = HasCollision # true if collision is detected # bool

# stores the details of a collision between two objects
class Collision:
    def __init__(self, ObjA, ObjB, Points):
        self.ObjA = ObjA # Object
        self.ObjB = ObjB # Object
        self.Points = Points # CollisionPoints


# collision response
# solve collisions
class Solver:
    def __init__(self):
        self.collisions = []
        

    def Solve(self, collisions, dt):
        pass


class PhysicsWorld:
    def __init__(self):
        self.objects = [] # list of objects
        self.gravity = Vector3(0., -9.81, 0.) # gravity vector
    
    def add_object(self, object):
        self.objects.append(object)
    
    def remove_object(self, object):
        self.objects.remove(object)
    
    # update the position and velocity based on forces acting on it
    def apply_forces(self, dt):
        for obj in self.objects:
            obj.Force += obj.Mass * self.gravity # apply a force
            
            obj.Velocity += obj.Force / obj.Mass * dt
            obj.Position += obj.Velocity * dt
            
            obj.Force = Vector3(0., 0., 0.) # reset net force at the end

class CollisionWorld(PhysicsWorld):
    def init(self):
        super().__init__()
        self.solvers = [] # List of solvers to use when solving collisions
        self.callback_on_collision = None # Callback function which will be called when a collision occurs

    def add_collision_object(self, object):
        super().add_object(object)

    def remove_collision_object(self, object):
        super().remove_object(object)

    def add_solver(self, solver):
        self.solvers.append(solver)

    def remove_solver(self, solver):
        self.solvers.remove(solver)

    def set_collision_callback(self, callback):
        """Set the function to be called when a collision occurs."""
        self.callback_on_collision = callback

    def solve_collisions(self, collisions, dt):
        """Solve the given collisions using the solvers in the world."""
        for solver in self.solvers:
            solver.solve(collisions, dt)

    def send_collision_callbacks(self, collisions, dt):
        """Send the collision callbacks for the given collisions."""
        for collision in collisions:
            if self.callback_on_collision:
                self.callback_on_collision(collision, dt)
            if collision.ObjA.on_collision:
                collision.ObjA.on_collision(collision, dt)
            if collision.ObjB.on_collision:
                collision.ObjB.on_collision(collision, dt)

    def resolve_collisions(self, dt):
        # Find all collisions
        collisions = []
        triggers = []
        for a in self.objects:
            for b in self.objects:
                if a == b:
                    continue
                if not a.Collider or not b.Collider:
                    continue
                points = a.Collider.TestCollision(
                    a.Transform, b.Collider, b.Transform)
                if points.has_collision: # or if points:
                    if a.is_trigger() or b.is_trigger():
                        triggers.append(Collision(a, b, points))
                    else:
                        collisions.append(Collision(a, b, points))
        self.solve_collisions(collisions, dt) # Don't solve triggers
        self.send_collision_callbacks(collisions, dt)
        self.send_collision_callbacks(triggers, dt)


# Collision detection


# derived class of collider for sphere-sphere and sphere-plane collisions
class SphereCollider:   
    def __init__(self, Center=None, Radius=None):
        self.Center = Center
        self.Radius = Radius
    
    def TestCollision(self, transform, collideobject, collideobjectTransform):
        if isinstance(collideobject, SphereCollider):
            return algo.FindSphereSphereCollisionPoints(
                self, transform, collideobject, collideobjectTransform)
        elif isinstance(collideobject, PlaneCollider):
            return algo.FindSpherePlaneCollisionPoints(
                self, transform, collideobject, collideobjectTransform)
        elif isinstance(collideobject, Collider):
            return collideobject.TestCollision(collideobjectTransform, self, transform)


#Box, Capsule, Cylinder, Cone, ConvexMesh, ConcaveMesh, Compound, ConvexHull, StaticCollision class in the .cpp file.

# derived class of collider for plane-sphere and plane-plane collisions
class PlaneCollider:
    def __init__(self, Plane=None, Distance=None):
        self.Plane = Plane # Vector3
        self.Distance = Distance # float
    
    def TestCollision(self, transform, collideobject, collideobjectTransform):
        if isinstance(collideobject, SphereCollider):
            points = collideobject.TestCollision(collideobjectTransform, self, transform)
            # swap points because the normal of the plane is pointing outwards
            T = points.A # You could have an algo Plane v Sphere to do the swap
            points.A = points.B
            points.B = T
            points.Normal = -points.Normal
            return points
        elif isinstance(collideobject, PlaneCollider):
            return None # No plane v plane
        elif isinstance(collideobject, Collider):
            return collideobject.TestCollision(collideobjectTransform, self, transform)


# abstract base class that define pure virtual functions for ts+esting collisions with other colliders
# takes in transform object of the collider that is being tested abd returb a collisionpoints object to return the details of the collision
# can define derived class for specific types of colliders and override the testcollision function for that type of collider
class Collider:
    def __init__(self, *args, **kwargs):
        self.sphere_collider = SphereCollider(*args, **kwargs)
        self.plane_collider = PlaneCollider(*args, **kwargs)

    def TestCollision(self, transform, collideobject, collideobjectTransform):
        if isinstance(collideobject, SphereCollider):
            return self.sphere_collider.TestCollision(transform, collideobject, collideobjectTransform)
        elif isinstance(collideobject, PlaneCollider):
            return self.plane_collider.TestCollision(transform, collideobject, collideobjectTransform)

"""
class ColliderFactory:
    def create_collider(self, collider_type, **kwargs):
        if collider_type == "Sphere":
            if 'center' in kwargs and 'radius' in kwargs:
                return SphereCollider(kwargs['center'], kwargs['radius'])
        elif collider_type == "Plane":
            if 'plane' in kwargs and 'distance' in kwargs:
                return PlaneCollider(kwargs['plane'], kwargs['distance'])
        elif collider_type == "Collider":
            if 'center' in kwargs and 'radius' in kwargs and 'plane' in kwargs and 'distance' in kwargs:
                return Collider(kwargs['center'], kwargs['radius'], kwargs['plane'], kwargs['distance'])

factory = ColliderFactory()
collider = factory.create_collider("Sphere", center=[0,0,0], radius=1)

"""

# the algo namespace includes two static methods for finding the collision points between a sphere and another sphere or a plane
# take as input the SphereCollider and PlaneCollider objects being tested, as well as their corresponding Transform objects, and return a CollisionPoints object indicating the details of the collision.
class algo:
    @staticmethod
    def FindSphereSphereCollisionPoints(a, ta, b, tb):
        pass
    
    # a = SphereCollider, ta = Transform, b = SphereCollider, tb = Transform
    
    @staticmethod
    def FindSpherePlaneCollisionPoints(a, ta, b, tb):
        pass

    # a = SphereCollider, ta = Transform, b = PlaneCollider, tb = Transform


# more options

class CollisionObject(object):
	def __init__(self, transform, collider, isTrigger, isDynamic, onCollision):
		self.m_transform = transform
		self.m_collider = collider
		self.m_isTrigger = isTrigger # If the object is a trigger
		self.m_isDynamic = isDynamic # If the object is dynamic
		self.m_onCollision = onCollision # Callback function which will be called when the object collides with another object

    # getters & setters, no setter for m_isDynamic


class Rigidbody(CollisionObject):
    def __init__(self, mass, takesGravity, staticFriction, dynamicFriction, restitution):
        self.gravity = Vector3(0, -9.81, 0) # Gravitational acceleration
        self.position = Vector3(0, 0, 0)  
        self.velocity = Vector3(0, 0, 0) # Velocity of the rigidbody
        self.force = Vector3(0, 0, 0) # Force applied to the rigidbody
        self.mass = mass
        self.takesGravity = takesGravity # If the rigidbody will take gravity from the world.
        self.m_staticFriction = staticFriction # Static friction coefficient
        self.m_dynamicFriction = dynamicFriction # Dynamic friction coefficient
        self.m_restitution = restitution # Elasticity of collisions (bounciness)
    
    def set_position(self, position):
        self.position = position
        
    def get_position(self):
        return self.position
    
    def set_velocity(self, velocity):
        self.velocity = velocity
        
    def get_velocity(self):
        return self.velocity
    
    def set_force(self, force):
        self.force = force
    
    def get_force(self):
        return self.force




class DynamicsWorld(CollisionWorld):
    def __init__(self):
        super().__init__()
        self.m_gravity = Vector3(0, -9.81, 0)
    
    def add_rigidbody(self, rigidbody):
        if rigidbody.takes_gravity():
            rigidbody.set_gravity(self.m_gravity)
        
        self.add_collision_object(rigidbody)
    
    def apply_gravity(self):
        for object in self.objects:
            if not object.is_dynamic():
                continue
            
            rigidbody = Rigidbody(object)
            rigidbody.apply_force(rigidbody.gravity() * rigidbody.mass())
    
    def move_objects(self, dt):
        for object in self.m_objects:
            if not object.is_dynamic():
                continue
            
            rigidbody = Rigidbody(object)
            
            vel = rigidbody.velocity() + rigidbody.force() / rigidbody.mass() * dt
            
            rigidbody.set_velocity(vel)
            
            pos = rigidbody.position() + rigidbody.velocity() * dt
            
            rigidbody.set_position(pos)
            
            rigidbody.set_force(Vector3(0, 0, 0))
    
    def step(self, dt):
        self.apply_gravity()
        self.resolve_collisions(dt)
        self.move_objects(dt)


class CollisionObject:
    def init(self):
        self.m_transform = Transform(Position=Vector3(0, 0, 0), Rotation=Quaternion(0, 0, 0, 1), Scale=Vector3(1, 1, 1))
        self.m_lastTransform = Transform(Position=Vector3(0, 0, 0), Rotation=Quaternion(0, 0, 0, 1), Scale=Vector3(1, 1, 1))
        self.m_collider = Collider()
        self.m_isTrigger = False
        self.m_isStatic = False
        self.m_isDynamic = False
        self.m_onCollision = lambda collision, dt: None
        # getters & setters for everything, no setter for isDynamic


class PhysicsSmoothStepSystem:
    def init(self):
        self.accumulator = 0.0

    def update(self):
        for entity in get_all_physics_entities():
            transform = entity.get(Transform)
            object = entity.get(CollisionObject)
            
            last_transform = object.last_transform()
            current_transform = object.transform()
            
            transform.position = lerp(
                last_transform.position,
                current_transform.position,
                self.accumulator / physics_update_rate()
            )
        
        self.accumulator += frame_delta_time()

    def physics_update(self):
        self.accumulator = 0.0


