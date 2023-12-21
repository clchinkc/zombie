import tkinter as tk
from threading import Thread

import numpy as np
import pygame


class Fluid:
    def __init__(self, density, numX, numY, h):
        self.density = density
        self.numX = numX + 2  # To account for boundaries
        self.numY = numY + 2  # To account for boundaries
        self.h = h
        self.u = np.zeros((self.numY, self.numX))  # x-component of velocity
        self.v = np.zeros((self.numY, self.numX))  # y-component of velocity
        self.prev_u = np.zeros_like(self.u)
        self.prev_v = np.zeros_like(self.v)
        self.d = np.zeros((self.numY, self.numX))  # density
        self.prev_d = np.zeros_like(self.d)
        self.obstacle = np.zeros((self.numY, self.numX), dtype=bool)

    def simulate(self, dt):
        self.prev_u, self.u = self.u, self.prev_u
        self.prev_v, self.v = self.v, self.prev_v
        self.prev_d, self.d = self.d, self.prev_d

        self.diffuse(1, self.prev_u, self.u, dt)
        self.diffuse(2, self.prev_v, self.v, dt)

        self.project(self.prev_u, self.prev_v, self.u, self.v)

        self.advect(1, self.u, self.prev_u, self.prev_u, self.prev_v, dt)
        self.advect(2, self.v, self.prev_v, self.prev_u, self.prev_v, dt)

        self.project(self.u, self.v, self.prev_u, self.prev_v)

        self.diffuse(0, self.prev_d, self.d, dt)
        self.advect(0, self.d, self.prev_d, self.u, self.v, dt)
        
        # Add density at the left edge to visualize the wind flow
        # where there is no obstacle.
        self.set_flow(velocity=1.0, density=20.0)

    def diffuse(self, b, x, x0, dt):
        a = dt * self.density * self.numX * self.numY
        self.lin_solve(b, x, x0, a, 1 + 4 * a)
        
    def lin_solve(self, b, x, x0, a, c):
        for _ in range(20):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[0:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, 0:-2] + x[1:-1, 2:])) / c
            self.set_bnd(b, x)

    def project(self, u, v, p, div):
        div[1:-1, 1:-1] = -0.5 * (u[2:, 1:-1] - u[0:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, 0:-2]) / self.h
        p[1:-1, 1:-1] = 0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1, 4)
        
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / self.h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / self.h
        self.set_bnd(1, u)
        self.set_bnd(2, v)
        
    def advect(self, b, d, d0, u, v, dt):
        dt0 = dt * self.numX
        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                x = i - dt0 * u[j, i]
                y = j - dt0 * v[j, i]
                if x < 0.5:
                    x = 0.5
                elif x > self.numX - 1.5:
                    x = self.numX - 1.5
                i0 = int(x)
                i1 = i0 + 1
                if y < 0.5:
                    y = 0.5
                elif y > self.numY - 1.5:
                    y = self.numY - 1.5
                j0 = int(y)
                j1 = j0 + 1
                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1
                d[j, i] = s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0]) + s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
        self.set_bnd(b, d)

    def set_bnd(self, b, x):
        # Reflective boundaries
        x[0, :] = -x[1, :]
        x[-1, :] = -x[-2, :]
        x[:, 0] = -x[:, 1]
        x[:, -1] = -x[:, -2]
        
        # Corners
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

        # Boundary conditions for the obstacle
        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                if self.obstacle[j, i]:
                    if b == 1:
                        x[j, i] = -x[j - 1, i]
                    elif b == 2:
                        x[j, i] = -x[j, i - 1]
                    else:
                        x[j, i] = 0

        # Boundary conditions for the velocity
        x[:, 1] = x[:, 2] if b == 1 else x[:, 1]  # For left boundary
        x[:, -2] = x[:, -3] if b == 1 else x[:, -2]  # For right boundary


    def set_obstacle(self, x_center, y_center, radius):
        # Create an array to mark the obstacle cells
        self.obstacle = np.zeros((self.numY, self.numX), dtype=bool)
        for i in range(self.numX):
            for j in range(self.numY):
                distance = np.sqrt((i - x_center) ** 2 + (j - y_center) ** 2)
                if distance < radius:
                    self.obstacle[j, i] = True
                    self.u[j, i] = 0
                    self.v[j, i] = 0

    def set_flow(self, velocity, density):
        # Set a continuous flow from left to right other than the obstacle in the middle of the domain
        for j in range(self.numY // 2 - 10, self.numY // 2 + 10):
            if not self.obstacle[j, 1]:
                self.u[j, :] = velocity  # inflow only on the left edge where there is no obstacle
                self.d[j, 1] = density  # adding density at the left edge to visualize the wind flow

class FluidApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Euler Fluid Simulation")

        self.setup_buttons()

        self.fluid = Fluid(density=1.0, numX=100, numY=100, h=1)
        
        self.dt = 0.1
        
        self.scale_factor = 8
        self.obstacle_radius = 10

        self.pygame_thread = Thread(target=self.init_pygame)
        self.pygame_thread.daemon = True
        self.pygame_thread.start()
        
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def setup_buttons(self):
        wind_tunnel_btn = tk.Button(self.root, text="Wind Tunnel", command=lambda: self.set_scene(1))
        wind_tunnel_btn.pack()
        # Add other buttons and checkboxes similarly

    def set_scene(self, scene_id):
        if scene_id == 1:
            self.fluid = Fluid(density=1.0, numX=200, numY=100, h=1)
            # Add an obstacle
            obstacle_x = self.fluid.numX // 4
            obstacle_y = self.fluid.numY // 2
            self.fluid.set_obstacle(obstacle_x, obstacle_y, self.obstacle_radius)

    def render_fluid(self, screen):
        screen.fill((0, 0, 0))  # Clear screen
        density_surf = pygame.Surface((self.fluid.numX * self.scale_factor, self.fluid.numY * self.scale_factor), pygame.SRCALPHA)
        # Render the fluid density and the obstacle
        for i in range(1, self.fluid.numX - 1):
            for j in range(1, self.fluid.numY - 1):
                if self.fluid.obstacle[j, i]:
                    pygame.draw.rect(density_surf, (255, 255, 255), (i * self.scale_factor, j * self.scale_factor, self.scale_factor, self.scale_factor))
                else:
                    density_value = self.fluid.d[j, i]
                    alpha_value = min(255, max(0, int(density_value * 255)))
                    pygame.draw.rect(density_surf, (alpha_value, alpha_value, alpha_value), (i * self.scale_factor, j * self.scale_factor, self.scale_factor, self.scale_factor))

        # Render velocity vectors (optional)
        for i in range(1, self.fluid.numX - 1, 2):  # Skip some cells for clarity
            for j in range(1, self.fluid.numY - 1, 2):
                u = self.fluid.u[j, i]
                v = self.fluid.v[j, i]
                if np.linalg.norm([u, v]) > 0.01:  # Only draw significant velocities
                    start_pos = (i * 8 + 4, j * 8 + 4)
                    end_pos = (start_pos[0] + u * 100, start_pos[1] + v * 100)
                    pygame.draw.line(density_surf, (0, 255, 0), start_pos, end_pos)

        screen.blit(density_surf, (0, 0))

    def init_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()

        mouse_down = False
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    self.shutdown()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_down = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_down = False
                elif event.type == pygame.MOUSEMOTION:
                    if mouse_down:
                        x, y = event.pos
                        obstacle_x = x // self.scale_factor
                        obstacle_y = y // self.scale_factor
                        self.fluid.set_obstacle(obstacle_x, obstacle_y, self.obstacle_radius)  # radius 10 as an example


            self.fluid.simulate(self.dt)
            self.render_fluid(screen)
            pygame.display.flip()
            clock.tick(60)

    def shutdown(self):
        self.running = False
        if pygame.get_init():
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.root.after(0, self.root.quit)

    def run(self):
        self.root.mainloop()
        self.pygame_thread.join()

if __name__ == "__main__":
    app = FluidApp()
    app.run()

"""
The "Steps of Reduction" refer to a conceptual framework used to simplify complex systems into more manageable models. This is often applied in fields such as physics or applied mathematics. The framework is as follows:

1. **Reversible Micro-model**: This initial stage involves a microscopic model of a system where all processes are time-reversible, capturing the intricate details at the smallest scale.

2. **First Irreversible Model**: The system transitions from the reversible micro-model to an irreversible one. This marks a shift from time-reversible processes to those where the symmetry of time is not maintained, which is a common phenomenon when moving from microscopic to macroscopic descriptions.

3. **Kinetic Equation (Boltzmann)**: After achieving the first irreversible model, the next level of simplification is reached through the kinetic equation, specifically the Boltzmann equation. This is a cornerstone of statistical mechanics and provides a statistical representation of a thermodynamic system that is out of equilibrium.

4. **Hydrodynamic Equation**: The final reduction leads to the hydrodynamic equations, which are used to describe fluid flow at the macroscopic level. These equations generally encompass the principles of fluid dynamics, including the equations of continuity, motion (Navier-Stokes), and energy conservation.

Integral to this framework are the conceptual transitions between stages:

- The change from a **Reversible micro-description to a Dissipative macro-description** questions how processes known for their time-reversibility at the microscopic level manifest as dissipative and irreversible at the macroscopic level, where phenomena such as friction and viscosity become apparent, leading to energy dissipation.

- Understanding **how a system with many degrees of freedom is reduced to a system with a few degrees of freedom** addresses the method by which complex systems, characterized by numerous variables, can be simplified into models with far fewer variables. This simplification often involves the identification of collective behaviors or emergent properties that can effectively encapsulate the system's macroscopic behavior. 

This framework illustrates the methodology of systematically reducing the complexity of a system to make it more understandable and to derive predictive macroscopic models from detailed microscopic descriptions.
"""