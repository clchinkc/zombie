"""
To implement a simulation of a person's activity during a zombie apocalypse at school, we would need to define several classes and functions to represent the different elements of the simulation.

First, we would need a Person class to represent each person in the simulation. This class would have attributes to track the person's location, state (alive, undead, or escaped), health, and any weapons or supplies they may have. It would also have methods to move the person on the grid and interact with other people and zombies.

Next, we would need a Zombie class to represent each zombie in the simulation. This class would have similar attributes and methods as the Person class, but would also include additional attributes and methods to simulate the behavior of a zombie (such as attacking living people and spreading the infection).

We would also need a School class to represent the layout of the school and track the locations of people and zombies on the grid. This class would have a two-dimensional array to represent the grid, with each cell containing a Person or Zombie object, or None if the cell is empty. The School class would also have methods to move people and zombies on the grid and update their states based on the rules of the simulation.

Finally, we would need a main simulate function that would set up the initial conditions of the simulation (such as the layout of the school, the number and distribution of people and zombies, and any weapons or supplies), and then run the simulation for a specified number of steps. This function would use the School class to move people and zombies on the grid and update their states, and could also include additional code to track and display the progress of the simulation.
"""

import heapq
import random


class Person:
    def __init__(self, location, state, health, weapons, supplies):
        self.location = location
        self.state = state
        self.health = health
        self.weapons = weapons
        self.supplies = supplies

    def move(self, direction):
        # Update person's location based on direction
        self.location[0] += direction[0]
        self.location[1] += direction[1]

    def attack(self, other):
        # Attack the other person with a weapon, if possible
        if self.weapons:
            weapon = random.choice(self.weapons)
            other.defend(weapon)

    def defend(self, weapon):
        # Defend against the attack and take damage, if possible
        if self.state == "alive":
            self.health -= weapon.damage
            if self.health <= 0:
                self.state = "undead"

    def escape(self):
        # Attempt to escape the school, if possible
        if self.location == (0, 0):  # Assume the escape point is at (0, 0)
            self.state = "escaped"


class Zombie:
    def __init__(self, location, health):
        self.location = location
        self.health = health

    def move(self, direction):
        # Update person's location based on direction
        self.location[0] += direction[0]
        self.location[1] += direction[1]

    def attack(self, person):
        # Simulate attack on a living person
        if person.state == "alive":
            person.defend(self)


class School:
    def __init__(self, layout):
        self.layout = layout  # 2D array representing school layout
        self.people = []  # List of Person objects
        self.zombies = []  # List of Zombie objects
        self.num_alive = 0  # Number of people alive in the school
        self.num_undead = 0  # Number of people turned into zombies
        self.num_escaped = 0  # Number of people escaped from the school

    def move_person(self, person, direction):
        # Move the person in the specified direction, if possible
        new_location = [0, 0]
        new_location[0] = person.location[0] + direction[0]
        new_location[1] = person.location[1] + direction[1]
        if new_location[0] >= 0 and new_location[0] < len(self.layout) and new_location[1] >= 0 and new_location[1] < len(self.layout[0]):
            person.location = new_location

    def move_zombie(self, zombie, direction):
        # Move the zombie in the specified direction, if possible
        new_location = [0, 0]
        new_location[0] = zombie.location[0] + direction[0]
        new_location[1] = zombie.location[1] + direction[1]
        if new_location[0] >= 0 and new_location[0] < len(self.layout) and new_location[1] >= 0 and new_location[1] < len(self.layout[0]):
            zombie.location = new_location

    def update_states(self):
        # Update the states of all people and zombies in the school
        for person in self.people:
            if person.state == "alive":
                # Check if the person has escaped
                person.escape()
                if person.state == "escaped":
                    self.num_escaped += 1
                    self.num_alive -= 1
            elif person.state == "undead":
                # Check if the person has died from a zombie attack
                if person.health <= 0:
                    self.people.remove(person)
                    self.num_undead -= 1

        for zombie in self.zombies:
            # Check if the zombie has turned a person into a zombie
            for person in self.people:
                if person.state == "alive" and person.location == zombie.location:
                    person.state = "undead"
                    person.health = 100
                    self.num_alive -= 1
                    self.num_undead += 1


def simulate(layout, num_steps):
    # Create a School object with the specified layout
    school = School(layout)

    # Add people and zombies to the school
    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == "P":
                # Add a person to the school
                person = Person((i, j), "alive", 100, [], [])
                school.people.append(person)
                school.num_alive += 1
            elif layout[i][j] == "Z":
                # Add a zombie to the school
                zombie = Zombie((i, j), 100)
                school.zombies.append(zombie)
                school.num_undead += 1

    # Simulate num_steps steps of the zombie apocalypse
    for step in range(num_steps):
        # Move each person and zombie in a random direction
        for person in school.people:
            if person.state == "alive":
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                school.move_person(person, direction)
        for zombie in school.zombies:
            direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            school.move_person(zombie, direction)

        # Update the states of all people and zombies in the school
        school.update_states()

        # Print the current status of the simulation
        print(
            f"Step {step+1}: {school.num_alive} alive, {school.num_undead} undead, {school.num_escaped} escaped")


def generate_layout(num_people, num_zombies, size):
    # Initialize the layout with empty cells
    layout = [[None for _ in range(size)] for _ in range(size)]

    # Surround the layout with walls
    for i in range(size):
        layout[i][0] = "#"
        layout[i][size - 1] = "#"
        layout[0][i] = "#"
        layout[size - 1][i] = "#"

    # Place the people and zombies at random locations on the grid
    for i in range(num_people):
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if layout[x][y] is None:
                layout[x][y] = "P"
                break
    for i in range(num_zombies):
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            if layout[x][y] is None:
                layout[x][y] = "Z"
                break

    return layout


layout = generate_layout(10, 5, 20)

simulate(layout, 10)

"""
There are many ways to implement more complex algorithms for controlling the movements of people and zombies in a simulation. Here are a few examples of more advanced approaches you could consider:

Pathfinding algorithms: You could use algorithms such as A* or Dijkstra's algorithm to allow people and zombies to intelligently navigate the school layout and avoid obstacles. This could be useful for simulating more realistic or strategic movements, such as people trying to find the best route to escape the school or zombies trying to track down nearby survivors.

Flocking algorithms: You could use algorithms such as Boids or Reynolds' steering behaviors to simulate the collective movements of groups of people or zombies. This could be used to model the behavior of crowds or hordes, and could help to create more realistic and dynamic simulations.

Machine learning algorithms: You could use machine learning techniques such as reinforcement learning or decision trees to train people and zombies to make more intelligent decisions about their movements. This could allow the simulation to adapt and improve over time, and could potentially enable the people and zombies to learn from their experiences and make more effective decisions in the future.
"""

"""
implement the A* algorithm inside the simulation for controlling the movements of people and zombies
"""


class Survivor:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.state = "alive"
        self.health = 100
        self.weapon = None
        
    # other methods and attributes as before...
    
class Zombie:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.state = "undead"
        self.health = 100
        
    # other methods and attributes as before...


class School:
    def __init__(self, layout):
        self.layout = layout  # 2D array representing school layout
        self.people = []  # list of Person objects
        self.zombies = []  # list of Zombie objects

    def find_path(self, start, end):
        """
        Find the shortest path from start to end using the A* algorithm.
        """
        start_pos = start
        end_pos = end

        # priority queue for storing unexplored nodes
        heap = []
        heapq.heappush(heap, (0, start_pos, []))

        visited = set()  # set of visited nodes

        while heap:
            cost, current_pos, path = heapq.heappop(heap)
            row, col = current_pos

            if current_pos == end_pos:
                return path

            if current_pos in visited:
                continue

            neighbors = self.find_neighbors(row, col)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                r, c = neighbor
                if self.layout[r][c] == "#":
                    continue

                # calculate cost of reaching this neighbor
                g = cost + 1  # movement cost
                h = self.manhattan_distance(
                    neighbor, end_pos)  # heuristic cost
                f = g + h  # total cost

                new_path = path + [neighbor]
                heapq.heappush(heap, (f, neighbor, new_path))
                visited.add(current_pos)

        return path

    def find_neighbors(self, row, col):
        """
        Find the valid neighbors of a given cell.
        """
        neighbors = []

        if row > 0:  # UP
            neighbors.append((row - 1, col))
        if row + 1 < len(self.layout):  # DOWN
            neighbors.append((row + 1, col))
        if col > 0:  # LEFT
            neighbors.append((row, col - 1))
        if col + 1 < len(self.layout[0]):  # RIGHT
            neighbors.append((row, col + 1))
        return neighbors

    def manhattan_distance(self, start, end):
        """
        Calculate the Manhattan distance between two points.
        """
        x1, y1 = start
        x2, y2 = end
        return abs(x1 - x2) + abs(y1 - y2)
      
    def get_adjacent_people(self, entity):
        """
        Get a list of people that are adjacent to the given entity (person or zombie).
        """
        adjacent_people = []
        for person in self.people:
            if abs(person.x - entity.x) <= 1 and abs(person.y - entity.y) <= 1:
                adjacent_people.append(person)
        return adjacent_people

    def get_zombie_destination(self, source):
        """
        Get the closest person for the zombie to move towards.
        """
        # Find the closest person
        distances = []
        for person in self.people:
            distance = self.manhattan_distance(
                (source.x, source.y), (person.x, person.y))
            distances.append((distance, person))
        distances.sort()
        closest_person = distances[0][1]

        # Find the shortest path to the closest person
        start = (source.x, source.y)
        end = (closest_person.x, closest_person.y)
        path = self.find_path(start, end)
        if path:
            return path[-1]
        else:
            return None

    def get_human_destination(self, source):
        """
        Get the destination for the source to move towards.
        """
        # Find the closest person
        distances_to_people = []
        for person in self.people:
            distance = self.manhattan_distance(
                (source.x, source.y), (person.x, person.y))
            distances_to_people.append((distance, person))
        distances_to_people.sort()
        closest_person = distances_to_people[0][1]

        # Find the closest zombie
        distances_to_zombies = []
        for zombie in self.zombies:
            distance = self.manhattan_distance(
                (source.x, source.y), (zombie.x, zombie.y))
            distances_to_zombies.append((distance, zombie))
        distances_to_zombies.sort()
        closest_zombie = distances_to_zombies[0][1]

        # If the closest person is closer than the closest zombie, move towards the person
        # Otherwise, move away from the zombie
        if distances_to_people[0][0] < distances_to_zombies[0][0]:
            # Find the shortest path to the closest person
            start = (source.x, source.y)
            end = (closest_person.x, closest_person.y)
            path = self.find_path(start, end)
            if path:
                return path[-1]
            else:
                return None
        else:
            # Find the shortest path away from the closest zombie
            start = (source.x, source.y)
            end = self.get_farthest_cell(
                start, (closest_zombie.x, closest_zombie.y))
            path = self.find_path(start, end)
            if path:
                return path[-1]
            else:
                return None

    def move_person(self, person):
        """
        Move a person towards the destination.
        """
        destination = self.get_human_destination(person)
        if destination:
            person.x, person.y = destination

    def move_zombie(self, zombie):
        """
        Move a zombie towards the destination.
        """
        destination = self.get_zombie_destination(zombie)
        if destination:
            zombie.x, zombie.y = destination

    def update_states(self):
        """
        Update the state of each person and zombie.
        """
        for person in self.people:
            destination = self.get_human_destination(person)
            if destination:
                # Move the person towards the destination
                person.x, person.y = destination
            else:
                # No destination found, move the person in a random direction
                person.move(random.choice([-1, 0, 1]),
                            random.choice([-1, 0, 1]))

        for zombie in self.zombies:
            # Check if the zombie is adjacent to a person
            adjacent_people = self.get_adjacent_people(zombie)
            if adjacent_people:
                # Attack the first adjacent person
                person = adjacent_people[0]
                zombie.attack(person)
                if person.health <= 0:
                    # Person is dead, turn them into a zombie
                    person.state = "undead"
                    self.zombies.append(person)
                    self.people.remove(person)
            else:
                # No people nearby, move the zombie towards the closest person
                destination = self.get_zombie_destination(zombie)
                if destination:
                    zombie.x, zombie.y = destination
                else:
                    # No destination found, move the zombie in a random direction
                    zombie.move(random.choice(
                        [-1, 0, 1]), random.choice([-1, 0, 1]))

