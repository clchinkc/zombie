"""
This is a program that uses the VPython library to create a simple gravitational N-body simulation. The program sets up three spheres to represent the Sun, Earth, and Jupiter, and then calculates the gravitational forces between them using Newton's law of gravitation. The simulation then updates the position and velocity of each sphere using the Euler-Cromer method, which is a simple numerical method for solving differential equations.

The program initializes the positions, velocities, and masses of the three spheres, and then enters a loop that runs for a specified duration of time (in this case, 10 seconds). Within the loop, the program calculates the gravitational forces between each pair of spheres, updates the forces and velocities of the spheres, and then updates their positions based on their new velocities.

The program uses the VPython sphere object to create the spheres, and sets their properties (such as position, radius, color, and trail) using the object's attributes. The vector object is used to represent three-dimensional vectors, and is used to store the positions, velocities, and forces of the spheres.

Overall, this program provides a simple example of how to use VPython to create a physics simulation, and how to use numerical methods to solve differential equations.
"""
"""
# n-body simulation

import vpython as vp

solar_mass = 1.988544e30
earth_mass = 5.97219e24 / solar_mass
jupiter_mass = 1898.13e24 / solar_mass

large_body = vp.sphere(
    pos=vp.vector(0, 0, 0),
    radius=0.1,
    color=vp.color.yellow,
)
large_body.m = 1
large_body.vel = vp.vector(0, 0, 0)
large_body.force = vp.vector(0, 0, 0)
large_body.name = "sol"

small_body = vp.sphere(
    pos=vp.vector(1, 0, 0),
    radius=0.05,
    color=vp.color.blue,
    make_trail=True,
    # retain=100,
)
small_body.m = earth_mass
small_body.vel = vp.vector(0, 2*vp.pi, 0)
small_body.force = vp.vector(0, 0, 0)
small_body.name = "earth"

jupiter = vp.sphere(
    pos=vp.vector(5.2, 0, 0),
    radius=0.07,
    color=vp.color.orange,
    make_trail=True,
)
jupiter.m = jupiter_mass
# jupiter.vel = vp.vector(0.036524, -3.6524, 0)
jupiter.vel = vp.vector(0, -2.75, 0)
jupiter.force = vp.vector(0, 0, 0)
jupiter.name = "jupiter"

bodies = [large_body, small_body, jupiter]
N = len(bodies)

dt = 0.005
t = 0
T = 10
G = 4*vp.pi*vp.pi

while t < T:
    vp.rate(50)

    # Reset forces 
    for i in range(0, N):
        bodies[i].force = vp.vector(0, 0, 0)

    # Compute forces
    for i in range(0, N):
        body_i = bodies[i]
        m_i = bodies[i].m
        r_i = body_i.pos
        
        for j in range(i + 1, N):
            body_j = bodies[j]
            m_j = bodies[j].m
            r_j = body_j.pos
            
            r = r_i - r_j 

            F = - ((G*m_i*m_j) / (r.mag2 * r.mag)) * r

            body_i.force += F
            body_j.force -= F 

    # Euler-Cromer    
    for i in range(0, N):
        a = bodies[i].force / bodies[i].m
        bodies[i].vel += a * dt
        bodies[i].pos += bodies[i].vel * dt

    t += dt
    """
    
# single pendulum

import numpy as np
import vpython as vp

initial_position = vp.vector(-4, 0, 0)
initial_velocity = vp.vector(0, -20, 0)

ball = vp.sphere(
    pos=initial_position,
    radius=0.5,
    color=vp.color.cyan,
    make_trail=True,
    # retain=50
)
ball.velocity = initial_velocity
ball.mass = 0.1

rod = vp.cylinder(
    pos=initial_position, axis=-ball.pos, radius=0.1
)


dt = 0.005
t = 0
T = 20
# k = 10
k = lambda t: np.sin(t)
d = 4
g = -9.81 * 0
drag_coefficient = 0.001 * 0

# A plot
kinetic = vp.gcurve(color=vp.color.blue)

while t < T:
    vp.rate(100)
    relative_displacement = rod.length - d
    ball_force = - k(t) * relative_displacement * ball.pos.norm()
    ball_force.y += g
    ball_force -= drag_coefficient * ball.velocity.mag**2 * ball.velocity.norm()
    acceleration = ball_force / ball.mass 
    ball.velocity += acceleration * dt
    kinetic_energy = 0.5 * ball.mass * ball.velocity.mag**2
    kinetic.plot(t, kinetic_energy)
    ball.pos += ball.velocity * dt
    rod.pos = ball.pos
    rod.axis = -ball.pos
    t += dt
