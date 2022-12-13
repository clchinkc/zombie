
import random

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


transition_matrix = [
  # Safe, Overrun, Noise
  [0.1, 0.4, 0.5], # Hide
  [0.7, 0.2, 0.1], # Fight
  [0.2, 0.4, 0.4]  # Run
]


# Define a list of events
events = ["Safe", "Overrun", "Noise"]

# Define a list of actions
actions = ["Hide", "Fight", "Run"]

# Define a function to select the next action based on the current state and event
def select_action(person, event, use_randomness=1):
    # If use_randomness is True, use the transition matrix and a random number to determine the next action
    if use_randomness:

        # Choose a random number between 0 and 1
        random_number = random.random()

        # Initialize the cumulative probability
        cumulative_probability = 0

        # Get the current state of the person
        current_state = person.state.get_status()

        # Find the index of the current state in the list of states
        state_index = actions.index(current_state)

        # Iterate over the transition probabilities for the current state
        for i, probability in enumerate(transition_matrix[state_index]):
            # Add the probability to the cumulative probability
            cumulative_probability += probability

            # Check if the random number is less than the cumulative probability
            if random_number < cumulative_probability:
                # Return the corresponding action
                return actions[i]

    # If use_randomness is False, use the transition matrix to directly determine the next action
    else:
        # Get the current state of the person
        current_state = person.state.get_status()

        # Find the index of the current state in the list of states
        state_index = actions.index(current_state)

        # Find the index of the current event in the list of events
        event_index = events.index(event)

        # Use the indices to look up the next action in the transition matrix
        next_action_index = transition_matrix[state_index][event_index]

        # Return the next action
        return actions[next_action_index]
    
    # If no action is selected, return None
    return None

# Define a function to select the next event
def select_next_event(person):
    # If the school is overrun, return "Overrun"
    if person.school.is_overrun:
        return "Overrun"

    # If the school is not overrun and there is noise, return "Overrun" with probability 0.1
    if not person.school.is_overrun and person.school.is_noise and random.random() < 0.1:
        return "Overrun"
    else:
        return "Noise"

    # If the person is fighting and there is noise, return "Overrun" with probability 0.5
    if person.state.is_fighting and person.school.is_noise and random.random() < 0.5:
        return "Overrun"
    else:
        return "Noise"

    # If the person is running and there is noise, return "Overrun" with probability 0.3
    if person.state.is_running and person.school.is_noise and random.random() < 0.3:
        return "Overrun"
    else:
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







