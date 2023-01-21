
import random
from dataclasses import dataclass

## Overall description


class Human:
    def __init__(self, id, position, health, strength, armour, speed, intelligence):
        self.id = id
        self.position = position
        self.health = health
        self.strength = strength
        self.armour = armour
        self.speed = speed
        self.intelligence = intelligence
        self.inventory = []
        self.weapon = None

    @property
    def defence(self):
        return self.armour*random.uniform(0.5, 1.5)

    def choose_action(self):
        # choose between move, attack, pick up item, use item
        # if low health, use item
        # elif there is enemy nearby, attack
        # elif there is item nearby, pick up
        # elif there is human nearby, trade
        # elif there is enough condition, craft item
        # else move
        pass
    
    """
    def choose_action(self, agent):
	neighbors = self.get_neighbors(agent)
	if isinstance(agent, Human):
		for neighbor in neighbors:
			if isinstance(neighbor, Zombie):
				return "attack"
		return "move"
	elif isinstance(agent, Zombie):
		for neighbor in neighbors:
			if isinstance(neighbor, Human):
				return "attack"
		return "move"
    """

    def move(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy

    def attack(self, enemy):
        damage = (self.strength+self.weapon.damage)*random.uniform(0.5, 1.5)
        damage -= enemy.defence
        enemy.health -= damage
    
    def pick_up(self, item):
        self.inventory.append(item)
    
    def trade(self, other):
        if other.weapon.damage < self.weapon.damage or self.armour < other.armour: # more clever conditions, based on intelligence, trade different items, like specific amount of food for weapon
            self.weapon, other.weapon = other.weapon, self.weapon
            self.armour, other.armour = other.armour, self.armour
            
    def craft(self, item):
        # if food count > 10, remove food, craft item
        pass
            
    def use_item(self, item):
        if isinstance(item, Weapon):
            self.weapon = item
            self.inventory.remove(item)
        elif isinstance(item, PowerUp):
            self.health += item.health
            self.strength += item.strength
            self.armour += item.armour
            self.speed += item.speed
            self.intelligence += item.intelligence
            self.inventory.remove(item)
            
    def direction(self, grid):
        # move away from zombies, move towards humans, with a degree of exploration when low of items and weapons
        # if there is no human or zombie nearby, move randomly
        # check if it is a valid move
        # return direction of the human
        pass

    def escape(self):
        # define the win condition
        pass

    def die(self):
        pass
    
    def print(self):
        print('Human id:', self.id)
        print('Position:', self.position)
        print('Health:', self.health)
        print('Strength:', self.strength)
        print('Armour:', self.armour)
        print('Speed:', self.speed)
        print('Intelligence:', self.intelligence)
        print('Inventory:')
        for item in self.inventory:
            print(item)
        print('Weapon:')
        if self.weapon:
            print(self.weapon)
        else:
            print('None')
        print()
        
    def __str__(self):
        return 'Human id: {}, Position: {}, Health: {}, Strength: {}, Armour: {}, Speed: {}, Intelligence: {}'\
            .format(self.id, self.position, self.health, self.strength, self.armour, self.speed, self.intelligence)
            
    def __repr__(self):
        return 'Human id: {}, Position: {}, Health: {}, Strength: {}, Armour: {}, Speed: {}, Intelligence: {}'\
            .format(self.id, self.position, self.health, self.strength, self.armour, self.speed, self.intelligence)

@dataclass
class Weapon:
    location: tuple
    damage: int
    
@dataclass
class PowerUp:
    location: tuple
    health: int
    strength: int
    armour: int
    speed: int
    intelligence: int


class Zombie:
    def __init__(self, id, position, health, strength, speed, armour, intelligence):
        self.id = id
        self.position = position
        self.health = health
        self.strength = strength
        self.speed = speed
        self.armour = armour
        self.intelligence = intelligence
        
    @property
    def defence(self):
        return self.armour*random.uniform(0.5, 1.5)

    def choose_action(self):
        # choose between move, attack
        # if there is enemy nearby, attack
        # else move
        pass
    
    """
    def choose_action(self, agent):
	neighbors = self.get_neighbors(agent)
	if isinstance(agent, Human):
		for neighbor in neighbors:
			if isinstance(neighbor, Zombie):
				return "attack"
		return "move"
	elif isinstance(agent, Zombie):
		for neighbor in neighbors:
			if isinstance(neighbor, Human):
				return "attack"
		return "move"
    """

    def move(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy

    def attack(self, enemy):
        damage = self.strength*random.uniform(0.5, 1.5)
        damage -= enemy.defence
        enemy.health -= damage
        
    def direction(self, grid):
        # move towards humans
        # if there is no human or zombie nearby, move randomly
        # check if it is a valid move
        # return direction of the zombie
        pass
    
    def check_valid_move(self, grid, dx, dy):
        # check if the move is valid
        if not (0 <= self.position[0]+dx < len(grid) and 0 <= self.position[1]+dy < len(grid[0])):
            return False
        if grid[self.position[0]+dx][self.position[1]+dy] != None:
            return False
        return True

    def die(self):
        pass


class Environment:
    def __init__(self, m, n, item):
        self.grid = [[None for _ in range(m)] for _ in range(n)] # 2D list of None (empty), Human, Zombie
        # list of randomly assigned location of human
        self.humans = [(random.randint(0, m-1), random.randint(0, n-1)) for _ in range(10)]
        # list of randomly assigned location of zombie
        self.zombies = [(random.randint(0, m-1), random.randint(0, n-1)) for _ in range(10)]
        self.item = item # list of location of weapon, power-up, other resources
        
    def get_item(self, location):
        if location in self.item:
            return self.grid[location[0]][location[1]]
        
    def get_zombie(self, location):
        if location in self.zombies:
            return self.grid[location[0]][location[1]]
        
    def get_human(self, location):
        if location in self.humans:
            return self.grid[location[0]][location[1]]

    def add_human(self, location, human):
        self.humans.append(location)
        self.grid[location[0]][location[1]] = human

    def add_zombie(self, location, zombie):
        self.zombies.append(location)
        self.grid[location[0]][location[1]] = zombie
    
    def add_item(self, location, item):
        self.item.append(location)
        self.grid[location[0]][location[1]] = item

    def remove_human(self, location):
        if location in self.humans:
            self.humans.remove(location)
            self.grid[location[0]][location[1]] = None

    def remove_zombie(self, location):
        if location in self.zombies:
            self.zombies.remove(location)
            self.grid[location[0]][location[1]] = None
    
    def remove_item(self, location):
        if location in self.item:
            self.item.remove(location)
            self.grid[location[0]][location[1]] = None

    def update(self, epoch):
        self.initialize()
        for i in range(epoch):
            self.human_round()
            self.zombie_round()
            if self.end_condition():
                break
    
    def initialize(self):
        pass
    
    def human_round(self):
        for human_location in self.humans:
            human = self.get_human(human_location)
            action = human.choose_action()
            if action == "attack": # may change to enum
                for x in range(human.position[0]-1, human.position[0]+2): # may change to within certain range
                    for y in range(human.position[1]-1, human.position[1]+2):
                        if (x, y) in self.zombies:
                            zombie = self.get_zombie((x, y))
                            human.attack(zombie)
                            if zombie.health <= 0:
                                self.remove_zombie((x, y))
                            break
            elif action == "pick up":
                for x in range(human.position[0]-1, human.position[0]+2):
                    for y in range(human.position[1]-1, human.position[1]+2):
                        if (x, y) in self.item:
                            item = self.get_item((x, y))
                            human.pick_up(item)
                            self.remove_item((x, y))
                            break
                        
            elif action == "use":
                if not human.inventory:
                    continue
                item = human.inventory.pop()
                human.use_item(item)
                
            elif action == "move":
                dx, dy = human.choose_direction(self.grid)
                self.remove_human(human.position)
                human.move(dx, dy)
                self.add_human(human.position, human)
                
    def zombie_round(self):
        for zombie_location in self.zombies:
            zombie = self.get_zombie(zombie_location)
            action = zombie.choose_action()
            if action == "attack":
                for x in range(zombie.position[0]-1, zombie.position[0]+2):
                    for y in range(zombie.position[1]-1, zombie.position[1]+2):
                        if (x, y) in self.humans:
                            human = self.get_human((x, y))
                            zombie.attack(human)
                            if human.health <= 0:
                                self.remove_human(human_location)
                                zombie = Zombie()
                                self.add_zombie(human_location, zombie)
                            break
            elif action == "move":
                dx, dy = zombie.choose_direction(self.grid)
                self.remove_zombie(zombie.position)
                zombie.move(dx, dy)
                self.add_zombie(zombie.position, zombie)

    """
	def update(self):
		for agent in self.population:    
			action = self.choose_action(agent)
			if action == "move":
				direction = self.choose_direction(agent)
				self.move_agent(agent, direction)
			elif action == "attack":
				self.attack_neighbors(agent)
			else:
				continue
	"""

    def end_condition(self):
        pass

    def print_grid(self):
        for row in self.grid:
            for col in row:
                print(col, end=' ') # None -> '.', Human -> 'H', Zombie -> 'Z'
            print()
        print()

    def print_humans(self):
        pass

    def print_zombies(self):
        pass


class Simulation:
    def __init__(self, m, n, item):
        self.env = Environment(m, n, item)
        
    def run(self):
        # update
        # print
        pass

    def plot(self):
        pass


## Advanced

# power-ups or special abilities item
# item and other resources and weapon may use same probability pickup function

# use one grid for item, one grid for weapon, one grid for other resources, one grid for human, one grid for zombie
# so that we can use 1 and 0 to represent whether there is an item or not and use numpy to do matrix operation

# human may attack human

# cure to heal zombie to human or fight off infection else immediately turn into zombie

# Cellular Automata
# Conway's Game of Life
# fewer than 2 neighbors or more than 3 neighbors, die
# exactly neighbors, stay the same
# exactly 3 neighbors, become alive
# Predator-Prey
# surrounded by zombies, alive turn into zombie
# surrounded by humans, zombie dies

# smart weapon selection
# own a list of weapons with different damage and attack range
# if zombie is close, use melee weapon
# if zombie is far, use ranged weapon
# if zombie is far and no ranged weapon, use melee weapon
# if zombie is close and no melee weapon, use ranged weapon
# choose weapon according to distance and then damage
# each weapon has a different attack method (use strategy pattern)

# velocity and acceleration for human and zombie


# speed controls who attacks first, if dead can't attack back
# or not in turn-based game, attack in interval of speed time after encountering (use threading)

# add wall class, human and zombie can't move through wall, may place barrier to create complex grid
# add door class, only human can move through door
# add barricade class, human may place barricade which has health and can be destroyed by huamn and zombie
