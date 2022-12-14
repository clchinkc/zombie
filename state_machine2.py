
import random
import numpy as np

class School:
    def __init__(self):
        self.is_safe = False
        self.is_overrun = False
        self.is_noise = False

    def get_status(self):
        if self.is_overrun:
            return "overrun"
        elif self.is_noise:
            return "noisy"
        elif self.is_safe:
            return "safe"
        else:
            return "unknown"

    def set_safe(self):
        self.is_overrun = False
        self.is_noise = False
        self.is_safe = True

    def set_overrun(self):
        self.is_safe = False
        self.is_noise = False
        self.is_overrun = True

    def set_noise(self):
        self.is_safe = False
        self.is_overrun = False
        self.is_noise = True

class State:
    def __init__(self):
        self.is_hiding = False
        self.is_fighting = False
        self.is_running = False

    def get_status(self):
        if self.is_hiding:
            return "hiding"
        elif self.is_fighting:
            return "fighting"
        elif self.is_running:
            return "running"
        else:
            return "unknown"

    def set_hiding(self):
        self.is_hiding = True
        self.is_fighting = False
        self.is_running = False

    def set_fighting(self):
        self.is_hiding = False
        self.is_fighting = True
        self.is_running = False

    def set_running(self):
        self.is_hiding = False
        self.is_fighting = False
        self.is_running = True

class Person:
  def __init__(self, name):
    self.name = name
    self.school = School()
    self.state = State()

  def hide(self):
    self.state.set_hiding()

  def fight(self):
    self.state.set_fighting()

  def run(self):
    self.state.set_running()

  def set_school_safe(self):
    self.school.set_safe()

  def set_school_overrun(self):
    self.school.set_overrun()

  def set_school_noise(self):
    self.school.set_noise()

  def get_status(self):
    return self.name + " is currently " + self.state.get_status() + " and the school is " + self.school.get_status()

# Define a probability transition matrix of state and events to actions
transition_matrix = np.array([
    # Hide, Fight, Run
    [[0.11, 0.21, 0.31],    # Noise, hiding
     [0.41, 0.51, 0.61],    # Overrun, hiding
     [0.71, 0.81, 0.91]],   # Safe, hiding
    [[0.12, 0.22, 0.32],    # Noise, fighting
     [0.42, 0.52, 0.62],    # Overrun, fighting
     [0.72, 0.82, 0.92]],   # Safe, fighting
    [[0.13, 0.23, 0.33],    # Noise, running
     [0.43, 0.53, 0.63],    # Overrun, running
     [0.73, 0.83, 0.93]],   # Safe, running
])

"""
# use neural network in the action selection process
def select_action(person, event):
    # Create a list of the current state and event
    state_event = [person.state.is_hiding, person.state.is_fighting, person.state.is_running, event == "Safe", event == "Overrun", event == "Noise"]

    # Create a list of the probabilities of selecting each action
    probabilities = nn.predict(state_event)

    # Select the next action based on the probabilities
    action = np.random.choice(actions, p=probabilities)

    # Return the next action
    return action

# training of the neural network using the simulation function
def train_nn(nn, steps, epochs):
    # Iterate over the number of epochs
    for _ in range(epochs):
        # Generate a person
        person = Person()

        # Simulate the person's behavior
        actions_taken = simulate(person, steps)

        # Create a list of the current states and events
        state_events = []
        for i in range(len(actions_taken)):
            state_events.append([person.state.is_hiding, person.state.is_fighting, person.state.is_running, actions_taken[i] == "Safe", actions_taken[i] == "Overrun", actions_taken[i] == "Noise"])

        # Create a list of the next states and events
        next_state_events = []
        for i in range(len(actions_taken)):
            next_state_events.append([person.state.is_hiding, person.state.is_fighting, person.state.is_running, actions_taken[i] == "Safe", actions_taken[i] == "Overrun", actions_taken[i] == "Noise"])

        # Train the neural network
        nn.train(state_events, next_state_events)

    return True
"""

# Define a function to select the next action based on the current state and event
def select_action(person, event, use_randomness=1):
    # Get the person's current state
    state_index = actions.index(person.state.get_status())

    # Get the person's current school status
    event_index = events.index(event)

    # Set the transition probabilities for the current state and event
    transition_probs = transition_matrix[state_index, event_index]
    # If we are using randomness, then select the next action randomly
    if use_randomness:
        # Select the next action randomly based on the transition probabilities
        return np.random.choice(actions, p=transition_probs)
    # Otherwise, select the next action based on the maximum probability
    else:
        # Select the next action based on the maximum probability
        return actions[np.argmax(transition_probs)]

# Define a list of events
events = ["Safe", "Overrun", "Noise"]

# Define a list of actions
actions = ["Hide", "Fight", "Run"]

# Define a function to select the next event
def select_next_event(person):
    # If the school is overrun, return "Overrun"
    if person.school.is_overrun:
        return "Overrun"

    # If the school is not overrun and there is noise, return "Overrun" with probability 0.1
    if not person.school.is_overrun and person.school.is_noise and random.random() < 0.1:
        return "Overrun"
    elif not person.school.is_overrun and person.school.is_noise:
        return "Noise"

    # If the person is fighting and there is noise, return "Overrun" with probability 0.5
    if person.state.is_fighting and person.school.is_noise and random.random() < 0.5:
        return "Overrun"
    elif person.state.is_fighting and person.school.is_noise:
        return "Noise"

    # If the person is running and there is noise, return "Overrun" with probability 0.3
    if person.state.is_running and person.school.is_noise and random.random() < 0.3:
        return "Overrun"
    elif person.state.is_running and person.school.is_noise:
        return "Noise"

    # If the school is safe, return "Safe"
    if person.school.is_safe:
        return "Safe"

    # If none of the above conditions are met, return "Unknown"
    return "Unknown"


# Define a function to simulate the person's behavior
def simulate(person, steps, use_randomness=1):
    # Initialize a list to store the actions taken by the person
    actions_taken = []

    # Iterate over the number of steps
    for _ in range(steps):
        # Select the next event
        event = select_next_event(person)
        if event == "Safe":
            person.set_school_safe()
        elif event == "Overrun":
            person.set_school_overrun()
        elif event == "Noise":
            person.set_school_noise()
        else:
            raise RuntimeError

        # Select the next action based on the current state and event
        action = select_action(person, event, use_randomness)

        # Update the person's state and school based on the selected action
        if action == "Hide":
            person.hide()
        elif action == "Fight":
            person.fight()
        elif action == "Run":
            person.run()
        else:
            raise RuntimeError


        # Add the action to the list of actions taken
        actions_taken.append(action)

    # Return the list of actions taken
    return actions_taken
        
#Create a person
person = Person("John")

#Simulate the person's behavior
process = simulate(person, 10)
print(process)

"""
In a real-world scenario, it would be important to consider many additional factors and details
when simulating a person's behavior during a zombie apocalypse at school.
For example, the simulation could include information about the person's physical abilities,
the layout of the school and surrounding area, the availability of supplies and weapons,
and the movements and actions of other survivors and zombies.
Additionally, the simulation could include more complex behaviors and decision-making processes
to make the person's actions more realistic and unpredictable.
Furthermore, the simulation could be extended to include multiple people
and simulate their interactions and group dynamics.
Overall, a more detailed and realistic simulation would require a significant amount of planning
and development to accurately represent the complex and dynamic situation of a zombie apocalypse at school.
"""
# Person class include information about the person's physical abilities
class Person:
    def __init__(self, name, health, speed, strength):
        self.name = name
        self.health = health
        self.speed = speed
        self.strength = strength
        self.state = State()
        self.school = School()

    def hide(self):
        self.state.set_hide()

    def fight(self):
        self.state.set_fight()

    def run(self):
        self.state.set_run()

    def set_school_safe(self):
        self.school.set_safe()

    def set_school_overrun(self):
        self.school.set_overrun()

    def set_school_noise(self):
        self.school.set_noise()

    def get_status(self):
        return self.name + " is currently " + self.state.get_status() + " and the school is " + self.school.get_status()

# 2D map

# map information about the layout of the school and surrounding area, the availability of supplies and weapons, and the movements and actions of other survivors and zombies.
class Map:
    def __init__(self, size, supplies, weapons, survivors, zombies):
        self.layout = None
        self.supplies = supplies
        self.weapons = weapons
        self.survivors = survivors
        self.zombies = zombies

    def set_layout(self, layout):
        self.layout = np.random.choice(["Hallway", "Classroom", "Gym", "Cafeteria", "Library", "Auditorium", "Office", "Bathroom", "Locker Room", "Playground", "Parking Lot", "Field"], size=self.size)

    def get_status(self):
        return "There are " + str(self.supplies) + " supplies, " + str(self.weapons) + " weapons, " + str(self.survivors) + " survivors, and " + str(self.zombies) + " zombies."
