
# Natural Selection

"""
In a simulation of a zombie apocalypse, natural selection can be applied to the population of both humans and zombies, providing a basis for understanding how populations of zombies and humans might evolve over time in response to changing environmental conditions and selective pressures. For example:
The virus that causes zombies could evolve over time to become more or less virulent, or to affect different hosts in different ways. This evolution would be shaped by natural selection, as the most successful strains of the virus would be those that are able to infect and reproduce effectively.
The competition for resources between humans and zombies could lead to the evolution of adaptations that allow one species to outcompete the other. For example, humans might evolve to be faster or more efficient at finding food, or zombies might evolve to be better at detecting and attacking their prey.
Natural selection could also play a role in the development of resistance to the virus among survivors. Those who are immune to the virus would be more likely to survive and pass on their immunity to their offspring, while those who are not immune would be more likely to succumb to the infection. Over time, the proportion of the human population that is immune to the virus would increase.
By considering these factors, a simulation of a zombie apocalypse could provide a more nuanced and scientifically informed understanding of the dynamics of the outbreak and the likelihood of different outcomes.
"""
"""
Designing a natural selection simulation of a zombie apocalypse would require making a number of decisions about the specifics of the scenario, such as the initial conditions and the rules governing how humans and zombies interact. However, a possible approach could be as follows:
Start with a simple mathematical model for the populations of humans and zombies. This model could be a set of equations that describe how the populations change over time, taking into account factors such as births, deaths, and conversions from humans to zombies.
Add in parameters that describe the strengths of the various selective pressures acting on the populations. For example, the rate of zombie-human encounters, the probability of a human being infected and converted to a zombie, the mortality rate of zombies, and the mortality rate of humans.
Run the simulation over a period of time, starting with the initial conditions and updating the populations at regular intervals based on the mathematical model and the parameters.
Analyze the results of the simulation to see how the populations of humans and zombies evolve over time. This could include examining how the numbers of humans and zombies change, as well as how the parameters evolve over time.
Repeat the simulation with different parameters and initial conditions to explore the range of possible outcomes and to test the robustness of the results.
Refine the simulation as needed based on the results of the initial runs. This could include adjusting the parameters, adding or modifying equations, or incorporating additional factors that were not considered in the initial version of the simulation.
"""

# with: competition of resources between human and zombies
# without: predator and prey, unclear relationship between variables

# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid13.ipynb
# https://github.com/OwenDavisBower/Natural-Selection-Simulator-Python-/blob/master/natural_selection.py
# https://github.com/topics/natural-selection
# https://thekidshouldseethis.com/post/simulating-natural-selection
# https://pycon.org.il/2016/static/sessions/yoav-ram.pdf
# https://www.youtube.com/watch?v=r_It_X7v-1E
# https://github.com/SebLague/Ecosystem-2/tree/master
# https://github.com/Emergent-Behaviors-in-Biology/community-simulator/blob/master/Tutorial.ipynb
# 「空間落遲模型」(spatial lag model, SLM)

"""
The spatiotemporal change for a species over time can be written as
∂ρ/∂t = D∇^2ρ + α(r)ρ - βρ^2
---
where
ρ(r,t) is the density (in numbers) of a species at position r and time t
ρ(r,0), the distribution of species at the initial time
---
D∇^2ρ = diffusion of a species (i.e. a group of animals spreading out over some biome)
D = diffusion rate
d1 and d2 is the diffusion rate
∇^2ρ(r) > 0 means average density around r is more dense
∇^2ρ(r) < 0 means average density around r is less dense
---
α(r)ρ = growth of the species with α(r) as the abundance of resources at point r
α = growth rate
---
βρ^2 = death of the species, through natural causes and competition
β = tolerance constant
---
when multiplying by β, one obtains this equation (all parameters follow from above):
∂u/∂t = D∇^2u + α(r)u - u^2
and in one dimension:
∂u/∂t = d(∂^2u/∂x^2) + u(r(x) - u) letting r(x) be the resources given in time x
---
Now suppose there are two species with normalized densities u and v. The equation above can be adjusted by adding two new values a and b: the relative competing strengths of each species. The equations are given by
∂u/∂t = d1(∂^2u/∂x^2) + u(r(x) - u - av)
∂v/∂t = d2(∂^2v/∂x^2) + v(r(x) - bu - v)
with initial conditions u(x,0) = f(x) and v(x,0) = g(x)
a and b is the species competitiveness
a < b means species 1 is better at outcompeting species 2
a > b means species 1 is worse at outcompeting species 2
---
The Question:
What makes one species better at outcompeting another species given a closed boundary with a zero flux condition? (i.e. finite area and animals cannot leave the area). Things to keep in mind
a and b
d1 and d2
f(x) and g(x)
---
The Finite Difference method
Instead of using dx and dt, we make a finite approximation with Δx and Δt, leading to
u(i,m+1) = u(i,m) + (d1Δt)/(Δx^2) (u(i,m+1) + u(i,m-1) - 2u(i,m)) + u(i,m)(r(i) - u(i,m) - av(i,m))Δt
v(i,m+1) = v(i,m) + (d2Δt)/(Δx^2) (v(i,m+1) + v(i,m-1) - 2v(i,m)) + v(i,m)(r(i) - bu(i,m) - v(i,m))Δt

i = position index
m = time index
"""

import matplotlib.pyplot as plt
import numba

# import libraries
import numpy as np
from matplotlib import animation, cm
from matplotlib.animation import PillowWriter

# set up initial conditions

times = np.linspace(0, 100, 100000)
x = np.linspace(-2, 2, 60)
p1 = np.zeros([len(times), len(x)])
p2 = np.zeros([len(times), len(x)])
r = np.exp(-x**2/0.1)
p1[0] = np.exp(-(x-1.2)**2/0.02)
p2[0] = np.exp(-(x+1.2)**2/0.02)

d1 = 0.01 # higher means diffuse faster
d2 = 0.1 # lower means diffuse slower
# if same competitiveness, lower diffusion rate will seem to lose but always catch from behind
a = 1 # higher means better competitiveness
b = 0.5 # lower means worse competitiveness
# high competitiveness always get more resources and grow faster
# equilibrium when same diffusion rate same competitiveness

dt = np.diff(times)[0]
dx = np.diff(x)[0]

plt.figure(figsize=(10,6))
plt.title("initial position", loc='center')
plt.plot(x, r, color='green', label='Resources')
plt.plot(x, p1[0], label='Species 1')
plt.plot(x, p2[0], label='Species 2')
plt.legend(loc='upper right')
plt.show()

print("dt/dx**2 =", dt/dx**2) # dt/dx**2 means how many times the species will diffuse in one time step


# solve the equation and get the solution

@numba.jit("UniTuple(f8[:,:], 2)(f8[:,:], f8[:,:], f8[:])", nopython=True, nogil=True)
def solve_pop(p1, p2, r):
    for t in range(0,len(times)-1):
        for i in range(0, len(p1[0])):
            # Insulated Boundary Conditions
            if i==0:
                deriv2_1 = 2*p1[t][i+1] - 2*p1[t][i]
                deriv2_2 = 2*p2[t][i+1] - 2*p2[t][i]
            elif i==len(p1[0])-1:
                deriv2_1 =  2*p1[t][i-1] - 2*p1[t][i]
                deriv2_2 =  2*p2[t][i-1] - 2*p2[t][i]
            else:
                deriv2_1 = p1[t][i+1] + p1[t][i-1] - 2*p1[t][i]
                deriv2_2 = p2[t][i+1] + p2[t][i-1] - 2*p2[t][i]
                
            p1[t+1][i] = p1[t][i] + d1*dt/dx**2 * deriv2_1 + \
                         p1[t][i] * dt * (r[i] - p1[t][i] - a*p2[t][i])
            p2[t+1][i] = p2[t][i] + d2*dt/dx**2 * deriv2_2 + \
                         p2[t][i] * dt * (r[i] - p2[t][i] - b*p1[t][i])
            
    return p1, p2

p1, p2 = solve_pop(p1, p2, r)


# plot the solution

i= 10 # time step
plt.figure(figsize=(14,8))
plt.plot(x,p1[i], lw=3, label='Species 1, $d_1=${:.3f}'.format(d1))
plt.plot(x,p2[i], lw=3, label='Species 2, $d_2=${:.3f}'.format(d2))
plt.plot(x,r, ls='--', label='Resources $r(x)$')
plt.legend(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.grid()
plt.show()

# animate the solution
num_frames = 200

def animate(i):
    # Calculate the index for p1 and p2 arrays
    index = int(i * len(times) / num_frames)
    line1.set_data(x, p1[index])
    line2.set_data(x, p2[index])
    line3.set_data(x, r)
    
# Equal aspect ratio Figure with black background and no axes.
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.grid()
line1, = ax.plot(x, p1[0], lw=3, label='Species 1, $d_1=${:.3f}'.format(d1))
line2, = ax.plot(x, p2[0], lw=3, label='Species 2, $d_2=${:.3f}'.format(d2))
line3, = ax.plot(x, r, ls='--', lw=3)
ax.legend(fontsize=20)

ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50)
plt.show()





"""
**Project Title: Integrating Differential Equations, Matrix Algebra, and Computer Simulations for a Comprehensive Zombie Apocalypse Model**

**Objective and Approach:**
The project aims to simulate a hypothetical zombie apocalypse scenario, deploying an integrated approach involving differential equations, matrix algebra, and computer simulations. As a programmer, your role is crucial in transforming complex mathematical models into practical and comprehensive computer simulations, thus enabling the detailed visualisation of the progression and spatial dynamics of a zombie apocalypse.

**Phase 1: Mathematical Modeling & System Analysis:**
The initial phase consists of the development and integration of mathematical models and system analysis:
- **Differential Equations for Dynamic Modelling:** Develop a system of differential equations to capture the continuous evolution of zombie and human populations. This includes birth rates, death rates, migration, zombie infection rates, and human interventions. Develop a comprehensive understanding of how different variables interact and influence the overall system dynamics.
- **Matrix Algebra for System Analysis:** Convert the differential equations into matrix form and implement matrix operations to enable long-term behavior analysis of system stability, convergence, critical thresholds and tipping points. This will inform the initial conditions and parameters for the subsequent simulation.
- **Inclusion of Human Behaviour and Interventions:** Expand the model to incorporate human decision-making, adaptive behaviours, and strategic interventions such as quarantine measures and military responses.
- **Incorporation of Spatial Variables:** Introduce spatial considerations to the model, accounting for geographical spread and movement of populations between regions. These variables impact transmission rates and add complexity to the model.
- **Stochastic Elements for Uncertainty:** Given the unpredictable nature of a zombie apocalypse, incorporate stochastic elements and random events into the model to reflect the inherent uncertainty in such a scenario.

**Phase 2: Simulation Implementation & Sensitivity Analysis:**
The second phase involves translating the mathematical model into a functional computer simulation and performing sensitivity analyses:
- **Simulation Development and Optimisation:** Using appropriate programming languages and libraries, develop a computer simulation based on the established mathematical models. Implement the initial conditions and parameters derived from the matrix algebra analysis to ensure accurate simulation results.
- **Scenario Exploration and Sensitivity Analysis:** Use the simulation to experiment with different scenarios and conduct sensitivity analyses, adjusting assumptions and parameters to understand their influence on the outbreak's progression.
- **Visualisation of Results:** Represent the simulation results through visually engaging tools such as graphs, charts, and interactive displays, facilitating clear insights into population dynamics over time and across geographical regions. Interpret the simulation outputs and compare them with predictions from differential equations and matrix algebra analysis.

**Phase 3: Performance Enhancement, Documentation & Reporting:**
The final phase involves performance enhancement, documentation of the project, and comprehensive reporting:
- **Robustness Analysis and Performance Enhancement:** Assess the robustness of the model and simulation, ensuring it can accommodate different edge cases and user inputs. Enhance the performance of the simulation for handling large-scale scenarios and for efficient execution.
- **Project Documentation:** Document the entire process, including the development and integration of mathematical models, simulation implementation, and result interpretation.
- **Comprehensive Report Generation:** Compile a detailed report summarising the project's findings, insights, and implications, encapsulating the dynamics of a potential zombie apocalypse scenario.

This integrated approach of differential equations, matrix algebra, and computer simulations offers a robust, accurate, and detailed depiction of a hypothetical zombie apocalypse, reflecting not only population dynamics over time but also spatial spread and human adaptive behaviours. It showcases the strength of this integrated approach in modelling complex and unpredictable scenarios.
"""

"""

Simulating a complex scenario like a zombie apocalypse can be achieved effectively by integrating two complementary population simulation methods - a differential equation-based method and a matrix-based method. Each of these models has unique strengths and understanding their functionality can help conceptualize a comprehensive simulation framework.

Differential Equation-Based Model: This method models population dynamics as a system of differential equations, representing rates of change in population sizes over time. In a zombie apocalypse, these could include the rate at which humans are turned into zombies, the rate at which zombies or humans are killed, and the natural birth and death rates of the human population. It offers time continuity, flexibility, and the ability to directly represent rates of change. However, it might get complex with multiple factors, and the assumption of continuous change might not be realistic for all events during a zombie apocalypse.

Matrix-Based Model: This method uses a matrix to describe probabilities of individuals transitioning between different states or stages in their life cycle. For a zombie apocalypse, you could have different states for humans (like uninfected, infected but not turned, turned into a zombie) and different events that cause transitions between states (like being bitten, dying of infection, turning into a zombie). It offers structured state transition, the ability to model discrete events, and can incorporate demographic structure to provide information of different granularity. However, discrete time steps might not fully capture the dynamics of a rapidly changing situation, static transition rates might not accurately represent the changing dynamics of a zombie apocalypse, and the future state of an individual is often assumed to be independent of its past.

Both methods can model growth, decline, or stability of populations over time, and both can incorporate stage- or age-structure. The key to integrating these methods is to identify aspects of population dynamics that can be represented in both models, such as survival rates, growth rates, and reproductive rates.

The strength of this integrated approach lies in the dynamic interplay between the matrix model and the differential equations. The rates calculated in the differential equations (such as the rate of zombie attacks or infection rates) could be adjusted based on the proportions of different states represented in the matrix model. For instance, if the matrix model shows a significant portion of the human population being elderly individuals less able to resist zombie attacks, this could be factored in as an increase in the attack rate in the differential equations. The contact matrix from the matrix model can influence the parameters of the differential equations, like the infection rate (β) and conversion rate (γ), based on the interactions between humans and zombies in different stages and locations.

At each time step, the matrix model updates the individual states based on these rates. These updated states then feed into the differential equations model, prompting an update in the overall populations. In the next cycle, the matrix model uses these updated population figures to modify the transition probabilities, resulting in a cyclical interaction between the two models. Meanwhile, the overall populations derived from the differential equations model could influence the state classifications and transitions within the matrix model, thus fostering a two-way relationship.

Identify the interactions between the macro-level variables in the differential equations model, the micro-level variables represented in the matrix algebra model, and the individual elements in the computer simulations. These interactions can form feedback loops, where the macro-level dynamics influence the simulation behaviors, and the results of the simulations, in turn, affect the macro-level variables.

Here's a general guide on how to integrate these approaches:

Stage 1 - Conceptualization: Conceptualize a combined model that accommodates the strengths of both approaches. This might involve representing some aspects of population dynamics as differential equations (e.g., overall population growth or decline) and others as a matrix (e.g., transitions between life stages).

Stage 2 - Implementation: Implement this combined model in a computational framework, which could involve developing custom code or using existing software tools that can handle both types of models.

Stage 3 - Data Exchange and Synchronization: Create mechanisms for data exchange and synchronization among the three models. The outputs of the differential equations and matrix algebra models can serve as inputs or guiding parameters for the computer simulations, and vice versa.

Stage 4 - Calibration and Validation: Calibrate the model using real-world data and validate it to ensure it accurately represents the dynamics of the population being studied. For discrete events, such as a large-scale zombie invasion or the discovery of a cure, these can be incorporated into the model by making suitable adjustments to the matrix or the differential equations at given moments in time.

Stage 5 - Model Additional Complexities: Next, expand your model to capture more complexities of the situation. Consider incorporating spatial dynamics by dividing the population into different geographical regions and modifying the differential equations to account for movement between these regions. You could also include stochastic elements to simulate random events, which could impact either the matrix or the differential equations.

Afterwards, visualize and analyze the results of the integrated model to gain insights into the system's behavior. Understand how the macro-level dynamics from the differential equations, the variables from the matrix algebra, and the individual-level interactions from the simulations contribute to the overall system behavior.

In a zombie apocalypse scenario, the overall population growth and decline could be modeled using a system of differential equations, while the state transitions (like uninfected to infected, infected to zombie) could be modeled using a matrix-based model. The two models could be linked by making the growth rate in the differential equation dependent on the stage distribution of the population as determined by the matrix model. Individual interactions and stochastic events could be modeled using computer simulations. The three models could be linked through feedback loops and data exchange mechanisms to create a comprehensive model of the zombie apocalypse.

In summary, this combined model captures both macro-level trends and micro-level individual factors, leading to a more robust and well-rounded understanding of the outcomes of a zombie apocalypse. The interaction between the two models allows for a more comprehensive and flexible model than either method could alone, ensuring a dynamic, adaptable, and intricate depiction of such a scenario. The integrated approach ensures a comprehensive and dynamic simulation of a zombie apocalypse, proficient at managing both continuous population changes and discrete events or state-dependent factors. It captures the overall population trends, courtesy of the differential equations model, and the detailed individual factors, thanks to the matrix model. As such, it addresses the limitations of both models: the static transition rates issue of the matrix model and the continuity assumption of the differential equations model.
"""
