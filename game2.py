class Person:
  def __init__(self, location, state, health, weapons, supplies):
    self.location = location
    self.state = state
    self.health = health
    self.weapons = weapons
    self.supplies = supplies
    
    def move(self, direction):
        # update person's location based on direction
        pass
    
    def interact(self, other):
        # simulate interaction with another person
        pass
  
    def attack(self, other):
        # Attack the other person with a weapon, if possible
        pass

    def defend(self, other):
        # Defend against the other person's attack, if possible
        pass

    def escape(self):
        # Attempt to escape the school, if possible
        pass
    
    
class Zombie:
  def __init__(self, location, health):
    self.location = location
    self.health = health
    
    def move(self, direction):
        # update zombie's location based on direction
        pass
    
    def attack(self, person):
        # simulate attack on a living person
        pass
    
    
class School:
    def init(self, width, height, layout):
        self.width = width
        self.height = height
        self.layout = layout # 2D array representing school layout
        self.people = [] # list of Person objects
        self.zombies = [] # list of Zombie objects
    
def move_person(self, person, dx, dy):
    # Check if the move is within the bounds of the school grid
    if 0 <= person.x + dx < self.width and 0 <= person.y + dy < self.height:
        # Check if the destination cell is empty
        if self.layout[person.x + dx][person.y + dy] is None:
            # Update the person's location
            person.move(dx, dy)
            # Update the layout grid
            self.layout[person.x][person.y] = person
            self.layout[person.x - dx][person.y - dy] = None

def move_zombie(self, zombie, dx, dy):
    # Check if the move is within the bounds of the school grid
    if 0 <= zombie elaborating on the move method, the code would look something like this:
    if 0 <= zombie.x + dx < self.width and 0 <= zombie.y + dy < self.height:
        # Check if the destination cell is empty
        if self.layout[zombie.x + dx][zombie.y + dy] is None:
            # Update the zombie's location
            zombie.move(dx, dy)
            # Update the layout grid
            self.layout[zombie.x][zombie.y] = zombie
            self.layout[zombie.x - dx][zombie.y - dy] = None

def update_states(self):
    # Update the state of each person and zombie based on the rules of the simulation
    for person in self.people:
        if person.state == "alive":
            # Check for nearby zombies and attack or defend
            pass
        elif person.state == "infected":
            # Check if the person has any weapons or supplies to fight off the infection
            # If not, the person becomes a zombie
            pass
    for zombie in self.zombies:
        # Check for nearby people to attack
        pass

    
def simulate(width, height, layout, num_steps):
  # create School object with specified width, height, and layout
  school = School(width, height, layout)
  
  # add people and zombies to school
  for i in range(num_people):
    # create Person object with random location, state, health, weapons, and supplies
    person = Person(...)
    # add person to school
    school.people.append(person)
  for i in range(num_zombies):
    # create Zombie object with random location and health
    zombie = Zombie(...)
    # add zombie to school
    school.zombies.append(zombie)
  
  # simulate num_steps steps of the zombie apocalypse
  for step in range(num_steps):
    # move people and zombies on the grid
    for person in school.people:
      school.move_person(person, dx, dy)
    for zombie in school.zombies:
      school.move_zombie(zombie, dx, dy)
      
    # update the states of people and zombies
    school.update_states()
