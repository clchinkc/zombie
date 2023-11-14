import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class PIDController:
    def __init__(self, kp, ki, kd, control_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.control_limit = control_limit
        self.integral = 0
        self.prev_measured_value = None  # Initialize with None

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt

        # Use change in measured value for derivative calculation
        if self.prev_measured_value is None:
            derivative = 0
        else:
            derivative = (measured_value - self.prev_measured_value) / dt

        self.prev_measured_value = measured_value  # Update the previous measured value

        control_signal = self.kp * error + self.ki * self.integral - self.kd * derivative  # Note the minus sign in kd term

        # Apply control limit if defined
        if self.control_limit is not None:
            control_signal = np.clip(control_signal, -self.control_limit, self.control_limit)

        return control_signal

class AdaptivePIDController:
    def __init__(self, kp, ki, kd, kp_adapt, ki_adapt, kd_adapt, control_limit=None):
        self.base_kp, self.base_ki, self.base_kd = kp, ki, kd
        self.kp_adapt, self.ki_adapt, self.kd_adapt = kp_adapt, ki_adapt, kd_adapt
        self.control_limit = control_limit
        self.integral = self.prev_error = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # Adaptive tuning of PID parameters based on error magnitude
        kp = self.base_kp + self.kp_adapt * np.abs(error)
        ki = self.base_ki + self.ki_adapt * np.abs(error)
        kd = self.base_kd + self.kd_adapt * np.abs(error)

        control_signal = kp * error + ki * self.integral + kd * derivative
        self.prev_error = error

        if self.control_limit is not None:
            control_signal = np.clip(control_signal, -self.control_limit, self.control_limit)
        return control_signal

def missile_model(x, y, altitude, control_signals, dt, disturbance):
    # Incorporating robust control by handling disturbances
    x += control_signals['x'] * dt + disturbance
    y += control_signals['y'] * dt + disturbance
    altitude += control_signals['z'] * dt - 9.81 * dt + disturbance
    return x, y, altitude

def random_disturbance(strength=0.1):
    return np.random.normal(0, strength)

# Simulation parameters
setpoint_altitude = 1000  # Desired altitude (meters)
setpoint_x = 100  # Desired x position (meters)
setpoint_y = 100  # Desired y position (meters)

current_x, current_y, current_altitude = 0, 0, 500  # Initial positions

pid_x = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_y = PIDController(kp=0.2, ki=0.02, kd=0.05, control_limit=50)
pid_z = PIDController(kp=0.6, ki=0.1, kd=0.1, control_limit=100)  # Altitude control
# pid_x = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
# pid_y = AdaptivePIDController(kp=0.2, ki=0.02, kd=0.05, kp_adapt=0.005, ki_adapt=0.001, kd_adapt=0.005, control_limit=50)
# pid_z = AdaptivePIDController(kp=0.6, ki=0.1, kd=0.2, kp_adapt=0.01, ki_adapt=0.002, kd_adapt=0.01, control_limit=100)  # Altitude control

time_step = 0.1
n_steps = 500
time_points = np.linspace(0, n_steps * time_step, n_steps)

# Resetting data structures for logging and plotting
x_positions, y_positions, altitudes = [], [], []
x_errors, y_errors, z_errors = [], [], []
control_inputs_x, control_inputs_y, control_inputs_z = [], [], []
velocities_x, velocities_y, velocities_z = [], [], []
accelerations_x, accelerations_y, accelerations_z = [], [], []
environmental_conditions = []

# Initial states for velocity calculation
prev_x, prev_y, prev_altitude = 0, 0, 500

# Simulation loop
for _ in time_points:
    disturbance = random_disturbance()
    environmental_conditions.append(disturbance)

    control_signal_x = pid_x.update(setpoint_x, current_x, time_step)
    control_signal_y = pid_y.update(setpoint_y, current_y, time_step)
    control_signal_z = pid_z.update(setpoint_altitude, current_altitude, time_step)

    control_inputs_x.append(control_signal_x)
    control_inputs_y.append(control_signal_y)
    control_inputs_z.append(control_signal_z)

    current_x, current_y, current_altitude = missile_model(current_x, current_y, current_altitude,
                                                            {'x': control_signal_x, 'y': control_signal_y,
                                                            'z': control_signal_z}, time_step, disturbance)

    x_positions.append(current_x)
    y_positions.append(current_y)
    altitudes.append(current_altitude)

    # Calculate velocity and acceleration
    velocity_x = (current_x - prev_x) / time_step
    velocity_y = (current_y - prev_y) / time_step
    velocity_z = (current_altitude - prev_altitude) / time_step
    velocities_x.append(velocity_x)
    velocities_y.append(velocity_y)
    velocities_z.append(velocity_z)

    acceleration_x = (velocity_x - (prev_x - prev_x) / time_step) / time_step
    acceleration_y = (velocity_y - (prev_y - prev_y) / time_step) / time_step
    acceleration_z = (velocity_z - (prev_altitude - prev_altitude) / time_step) / time_step
    accelerations_x.append(acceleration_x)
    accelerations_y.append(acceleration_y)
    accelerations_z.append(acceleration_z)

    prev_x, prev_y, prev_altitude = current_x, current_y, current_altitude

    x_errors.append(setpoint_x - current_x)
    y_errors.append(setpoint_y - current_y)
    z_errors.append(setpoint_altitude - current_altitude)

# 3D Plotting and additional plots
fig = plt.figure(figsize=(14, 10))

# Trajectory Plot
ax1 = fig.add_subplot(321, projection='3d')
ax1.plot(x_positions, y_positions, altitudes, label='Missile Trajectory')
ax1.scatter(setpoint_x, setpoint_y, setpoint_altitude, color='r', marker='o', label='Target')
ax1.set_xlabel('X Position (meters)')
ax1.set_ylabel('Y Position (meters)')
ax1.set_zlabel('Altitude (meters)')
ax1.set_title('3D Trajectory of Missile')
ax1.legend()

# Error Plot
ax2 = fig.add_subplot(322)
ax2.plot(time_points, x_errors, label='X Error')
ax2.plot(time_points, y_errors, label='Y Error')
ax2.plot(time_points, z_errors, label='Altitude Error')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Error (meters)')
ax2.set_title('Control Errors Over Time')
ax2.legend()

# Velocity Plot
ax3 = fig.add_subplot(323)
ax3.plot(time_points, velocities_x, label='X Velocity')
ax3.plot(time_points, velocities_y, label='Y Velocity')
ax3.plot(time_points, velocities_z, label='Z Velocity')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity Over Time')
ax3.legend()

# Acceleration Plot
ax4 = fig.add_subplot(324)
ax4.plot(time_points, accelerations_x, label='X Acceleration')
ax4.plot(time_points, accelerations_y, label='Y Acceleration')
ax4.plot(time_points, accelerations_z, label='Z Acceleration')
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Acceleration (m/s²)')
ax4.set_title('Acceleration Over Time')
ax4.legend()

# Control Input Plot
ax5 = fig.add_subplot(325)
ax5.plot(time_points, control_inputs_x, label='Control Input X')
ax5.plot(time_points, control_inputs_y, label='Control Input Y')
ax5.plot(time_points, control_inputs_z, label='Control Input Z')
ax5.set_xlabel('Time (seconds)')
ax5.set_ylabel('Control Input')
ax5.set_title('Control Inputs Over Time')
ax5.legend()

# Environmental Conditions Plot
ax6 = fig.add_subplot(326)
ax6.plot(time_points, environmental_conditions, label='Environmental Disturbance')
ax6.set_xlabel('Time (seconds)')
ax6.set_ylabel('Disturbance')
ax6.set_title('Environmental Conditions Over Time')
ax6.legend()

plt.tight_layout()
plt.show()


"""
What can a PID (Proportional-Integral-Derivative) controller do?

How to implement and visualize the use of PID algorithm in a python simulation?

How to use the PID algorithm in zombie apocalypse simulation, missile simulation, or stock price prediction?

Using a PID controller in various simulations like a zombie apocalypse, missile guidance, or stock price prediction requires a creative adaptation of the PID principles to the specific context of each scenario. Here's how you might approach each one:

### 1. Zombie Apocalypse Simulation

In a zombie apocalypse simulation, a PID controller might not be the most intuitive tool, but it can be used creatively. For example, if you're managing resources (like food, ammunition, or medicine), you could use a PID controller to balance the use of these resources over time.

- **Proportional:** Adjust resource allocation based on the current level of resources and immediate needs.
- **Integral:** Account for accumulated shortages or surpluses over time to avoid running out of resources.
- **Derivative:** Respond to the rate of change in resource levels, for example, if resources are being depleted rapidly.
"""

"""
Integrating a PID (Proportional-Integral-Derivative) controller in a zombie apocalypse simulation is a fascinating concept that can add a layer of complexity and realism to the simulation. Let's delve deeper into how each component of the PID controller can be utilized effectively in this context:

### Proportional Control (P)
- **Direct Response to Immediate Threats**: The proportional part of the controller can be used to respond directly and immediately to the current state of the simulation. For example, if the number of zombies within a certain area increases suddenly, the proportional control can immediately increase defensive measures such as fortifying barriers or dispatching more fighters to that area.
- **Resource Allocation**: This can also extend to resource management. If the survivor population increases, proportional control can increase the allocation of food and medical supplies accordingly.

### Integral Control (I)
- **Long-Term Strategy Adjustments**: The integral part accumulates the total error over time and responds based on this accumulated value. In the context of a zombie apocalypse, this could be used for long-term strategies such as expanding the safe zone or developing a cure. For instance, if the zombie population has been consistently above a certain threshold, the integral control could trigger research and development efforts for better weapons or defenses.
- **Recovery and Rebuilding Efforts**: The integral component could also manage the rebuilding of infrastructure or repopulation efforts, adjusting these activities based on the long-term trends in the simulation.

### Derivative Control (D)
- **Handling Sudden Changes**: The derivative part is particularly useful in reacting to rapid changes in the simulation. If there is a sudden outbreak or a massive wave of zombies, the derivative control can quickly mobilize emergency response measures.
- **Predictive Adjustments**: It can also be used for predictive adjustments. For instance, if the rate of zombie population increase is accelerating, the simulation can preemptively increase defense measures even before the situation becomes critical.

### Implementing PID in Simulation

1. **Define the Variables**: Identify the key variables in the simulation that need to be controlled, such as zombie population, survivor numbers, resource levels, and threat levels.

2. **Set the Objectives (Setpoints)**: Determine the desired state for these variables. For instance, keeping the zombie population below a certain number or maintaining a minimum level of resources for the survivors.

3. **Measure and Calculate Error**: Continuously monitor the current state and calculate the error, which is the difference between the current state and the setpoint.

4. **Apply the PID Formula**: Use the PID formula, where the control action is determined by the sum of the proportional, integral, and derivative responses. Fine-tuning the coefficients for P, I, and D is crucial for effective control.

5. **Execute Control Actions**: Based on the output from the PID controller, implement actions in the simulation. This could include adjusting the number of zombies introduced, changing the rate of resource consumption, or modifying the behavior of non-player characters (NPCs).

6. **Feedback and Tuning**: Continuously assess the effectiveness of the control actions and adjust the PID parameters for better results.

### Example Application
Consider a scenario where the primary objective is to prevent the overrun of a survivor base by zombies. The PID controller would work as follows:

- **P**: Increase guards and fortifications in response to an increase in nearby zombie numbers.
- **I**: Over time, as the threat persists, allocate more resources to long-term defenses and survivor health.
- **D**: If a sudden surge in zombies is detected, immediately deploy emergency measures like evacuation or calling for reinforcements.

In conclusion, integrating a PID controller into a zombie apocalypse simulation can significantly enhance the dynamic and adaptive nature of the simulation, making it more engaging and challenging. It requires careful consideration of how the simulation variables interact and how the PID controller's parameters should be tuned for optimal performance.
"""

"""
To implement a PID controller in a Python-based zombie apocalypse simulation, you will need to create a simulation environment where various factors like the number of zombies, number of survivors, resources, and threat levels are quantifiable and controllable. Let's outline a plan for this implementation:

Step 1: Define the Simulation Environment
Create a Simulation Grid: Design a grid-based map where each cell can contain zombies, survivors, resources, or be empty.
Initialize Variables: Define variables for the number of zombies, survivors, resources (food, medicine, weapons), and other relevant factors like health levels, fatigue, etc.
Set Simulation Rules: Establish rules for how zombies and survivors interact, how resources are consumed, and what conditions lead to changes (e.g., survivors turning into zombies, resource depletion).
Step 2: Set Up the PID Controller
Import a PID Library: Use a Python library like simple-pid for the PID controller functionality, or you can implement your own PID logic.
Define PID Parameters: Set the proportional (P), integral (I), and derivative (D) gains. These will need to be tuned during testing to achieve the desired behavior.
Determine Control Variables: Decide what aspects of the simulation the PID controller will adjust. This could be the rate of zombie generation, resource allocation, survivor recruitment, etc.
Step 3: Implementing the Control Loop
Integrate PID with the Simulation: At each simulation step (or time interval), calculate the current state of the environment (number of zombies, survivor status, resource levels).
Calculate Error: Determine the difference between the current state and your desired setpoints (e.g., ideal number of survivors, manageable number of zombies).
Apply PID Control: Use the PID controller to adjust your control variables based on the error. For instance, if there are too many zombies, the PID output might increase defensive actions or reduce zombie generation rate.
Step 4: Simulation Dynamics
Simulate Interactions: Define how zombies and survivors interact, how battles are fought, how survivors use resources, and how they move around the grid.
Resource Management: Implement logic for resource consumption, replenishment, and allocation based on PID output.
Event Handling: Program random events or specific triggers that can affect the simulation, like a horde of zombies appearing or a drop in resources.
Step 5: Testing and Tuning
Run Simulations: Execute the simulation and observe the outcomes. Look for stability, realistic behavior, and alignment with your objectives.
Tune PID Parameters: Adjust the P, I, and D gains based on the outcomes. The goal is to achieve a balance where the simulation responds effectively to changes without becoming unstable or oscillatory.
Step 6: Visualization and Analysis
Visual Output: Implement a way to visually represent the simulation, such as using matplotlib for plotting or a more interactive approach with a library like pygame.
Data Logging: Keep track of key metrics over time for analysis. This can help in understanding the long-term trends and the impact of PID control.
"""

"""
We can outline a comprehensive and advanced plan for creating a highly realistic and intricate zombie apocalypse simulation using Python. This simulation will integrate sophisticated behavior models, detailed interactions, complex decision-making logic, adaptive behaviors, advanced PID control mechanisms, and enhanced visualization and analytics.

### Comprehensive Plan for Advanced Zombie Apocalypse Simulation with PID Control

#### Step 1: Advanced Simulation Environment
- **Complex Grid System**: Develop a detailed grid with various terrain types such as urban areas, forests, and rural settings, affecting movement and encounters.
- **Dynamic Variables**: Incorporate variables like morale, weather conditions, time of day, and resource availability, influencing the behaviours and effectiveness of both zombies and survivors.
- **Resource Dynamics and Environmental Factors**: Incorporate diverse resources with specific uses, expiration dynamics, and environmental influences like terrain, weather, and day-night cycles.
- **Detailed Interaction Rules**: Establish complex rules for encounters, influenced by factors like survivors' skills, available resources, and the environment.

#### Step 2: Sophisticated Behavior Models and Decision-Making
- **State Machines or AI for Behaviors**: Implement state machines or AI algorithms for intricate models of zombie and survivor behaviors in different situations.
- **Adaptive Zombie Behavior**: Zombies exhibit varying behaviors such as forming hordes, being attracted to noise, or showing different aggression levels.
- **Strategic Survivor Behavior**: Enable survivors to make complex decisions, form alliances, build defenses, and adapt to changing conditions.

#### Step 3: Enhanced PID Controller Setup
- **Custom Multi-Input PID Controllers**: Design sophisticated PID controllers that handle multiple inputs and outputs, targeting various aspects of the simulation.
- **Adaptive PID Parameters**: Allow PID parameters to dynamically adapt to changes in the simulation, reflecting shifts in strategies or environmental conditions.
- **Control Diverse Aspects**: Use PID controllers to manage aspects like survivor recruitment, resource discovery, and zombie evolution.

#### Step 4: Integrated Control Loop with Advanced Dynamics
- **Real-Time State Evaluation**: Sophisticated evaluation system considering the compound effects of various factors.
- **Multi-Dimensional Error Assessment**: Implement advanced error calculation for comprehensive evaluation against desired outcomes.
- **Responsive PID Adjustment**: Dynamically adjust PID outputs to influence multiple aspects, ensuring a complex and responsive environment.

#### Step 5: Scalability and Enhanced Event System
- **Scalability Considerations**: Ensure the simulation can handle a large number of entities and interactions efficiently.
- **Enhanced Event System**: Develop a system for generating diverse events like internal conflicts, migrations, and environmental changes.

#### Step 6: In-Depth Testing, Tuning, and Analysis
- **Scenario-Based Testing**: Test under various scenarios to evaluate effectiveness and adaptability of PID controllers.
- **Automated Tuning and Machine Learning**: Employ machine learning for PID parameter tuning and pattern analysis.
- **Data Logging and Analysis Tools**: Implement comprehensive logging and develop tools for detailed analysis.

#### Step 7: Advanced Visualization and Interaction
- **Interactive Graphical Interface**: Use libraries like Pygame for an immersive graphical interface, allowing users to observe and potentially intervene in the simulation.
- **In-depth Analytics and Visualization**: Integrate tools for tracking and visualizing a wide range of metrics with advanced data analysis techniques.

By integrating these enhancements, the zombie apocalypse simulation will become an advanced tool for exploring complex systems dynamics, offering an engaging and insightful experience into the interplay of various factors in a simulated scenario. This comprehensive plan sets a foundation for developing a simulation that is not only realistic and challenging but also provides deep insights and analytics capabilities.
"""

"""
Your suggestions for refining the implementation of a PID controller in a Python-based zombie apocalypse simulation are insightful and would significantly enhance the simulation's realism and effectiveness. Let's break down how these modifications can be incorporated:

### Step 2: Set Up the PID Controller
- **Clear PID Objectives**: Define specific objectives for each PID controller, such as maintaining a desired survivor-to-zombie ratio or keeping resource levels within a certain range.
- **Multiple PID Controllers**: Implement different PID controllers for distinct aspects of the simulation, each with its own set of parameters tailored to control specific variables effectively.

### Step 3: Implementing the Control Loop
- **Nuanced Error Calculation**: Design the error calculation to reflect the complexity of the simulation. For example, if the goal is to maintain a survivor-to-zombie ratio, the error function could consider both the absolute numbers and their rate of change.
- **Context-Sensitive PID Output**: Make the PID output adaptive to the current simulation state. The same PID output might have different effects in varying scenarios, such as crisis versus stability.

### Step 4: Simulation Dynamics
- **Dynamic PID Influence**: Allow the impact of PID adjustments to vary based on the current state of the simulation. In critical situations, even minor PID adjustments could have significant effects.
- **Feedback Loop for Resources and Population**: Implement a mechanism where the state of resources and population health influences the effectiveness of the PID controller.

### Step 5: Testing and Tuning
- **Scenario-Based Testing**: Test the simulation under various conditions to ensure the PID controllers are robust and effective in different scenarios.
- **Performance Metrics**: Develop metrics to assess the PID controller's performance, focusing on its ability to stabilize the system and respond to disturbances.

### Step 6: Visualization and Analysis
- **Real-Time PID Visualization**: Incorporate real-time visualization of the PID controller’s output and its impact, such as graphs or indicators within the simulation.
- **Analysis Tools**: Add tools for in-depth analysis to compare the simulation's state under PID control versus without it in identical scenarios.

Implementing these modifications will make the PID controller a more integral and effective part of the simulation. The controller's responses will be more tailored to the specific needs of the simulation, providing a more realistic and dynamic experience. The addition of real-time visualization and analysis tools will also aid in understanding and fine-tuning the controller's impact.
"""
