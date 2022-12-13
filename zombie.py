"""
To implement a simulation of a person's activity during a zombie apocalypse at school, we would need to define several classes and functions to represent the different elements of the simulation.

First, we would need a Person class to represent each person in the simulation. This class would have attributes to track the person's location, state (alive, undead, or escaped), health, and any weapons or supplies they may have. It would also have methods to move the person on the grid and interact with other people and zombies.

Next, we would need a Zombie class to represent each zombie in the simulation. This class would have similar attributes and methods as the Person class, but would also include additional attributes and methods to simulate the behavior of a zombie (such as attacking living people and spreading the infection).

We would also need a School class to represent the layout of the school and track the locations of people and zombies on the grid. This class would have a two-dimensional array to represent the grid, with each cell containing a Person or Zombie object, or None if the cell is empty. The School class would also have methods to move people and zombies on the grid and update their states based on the rules of the simulation.

Finally, we would need a main simulate function that would set up the initial conditions of the simulation (such as the layout of the school, the number and distribution of people and zombies, and any weapons or supplies), and then run the simulation for a specified number of steps. This function would use the School class to move people and zombies on the grid and update their states, and could also include additional code to track and display the progress of the simulation.

class Person:
    def __init__(self, x, y, state, health, weapons, supplies):
        self.x = x
        self.y = y
        self.state = state
        self.health = health
        self.weapons = weapons
        self.supplies = supplies

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

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
    
  def attack(self, person):
    # simulate attack on a living person
    
    
class School:
  def __init__(self, layout):
    self.layout = layout  # 2D array representing school layout
    self.people = []  # list of Person objects
    self.zombies = []  # list of Zombie objects
    
  def move_person(self, person, direction):
    # move person in specified direction
    
  def move_zombie(self, zombie, direction):
    # move zombie in specified direction
    
  def update_states(self):
    # update state of each person and zombie based on rules of simulation
    
    
def simulate(layout, num_steps):
  # create School object with specified layout
  # add people and zombies to school
  # simulate num_steps steps of the zombie apocalypse
  

"""

from typing import Optional
import random


class Object:
    def __init__(self, row:int, column:int, name: Optional[str]=None):

        self.row, self.column = row, column
        self.name = name

class Character(Object):

    HEALTH = 0
    DAMAGE = 0
    AOE = 0

    def __init__(self, row:int, column:int, name:Optional[str]=None, health:int=-1, damage:int=-1, aoe:int=-1, food:int=100, *args, **kwargs):
        super().__init__(row, column, name)
        if health == -1:
          health = random.choice(self.HEALTH)
        self.health = health
        self.spawn_health = health

        if damage == -1:
          damage = random.choice(self.DAMAGE)
        self.damage = damage
        self.spawn_damage = damage

        if aoe == -1:
          aoe = random.choice(self.AOE)
        self.aoe = aoe
        self.spawn_aoe = aoe

        self.food = food

    # find new method (deterministic + random)
    # move speed to init
    def direction(self):
        move_row = 0
        move_column = 0
        move_speed = random.choice(range(-1,2))
        if(random.random() > 0.5):
            move_row = move_speed
        else:
            move_column = move_speed
        return move_row, move_column

    def move(self, destination_row, destination_column):
        if self.row != destination_row:
            self.food -=1
        if self.column != destination_column:
            self.food -=1
        self.row = destination_row
        self.column = destination_column

    def hit(self, damage:float):
        self.health -= damage
    
    def eat(self, food: Object):
        self.food += food.amount
    
    def starve(self):
        if self.food <= 0:
          self.health -= 1

    def heal(self, amount:float):
          self.health = min(self.health + amount, self.spawn_health)

    def __str__(self):
        if self.name:
          return self.name
        else:
          return self.__class__.__name__


class Human(Character):
    counter = 0

    HEALTH = range(10, 50)
    DAMAGE = range(1, 3)
    AOE = range(1, 2)

    # make use of character id and counter (faster calculation or import models)
    # separate count method
    def __init__(self, row:int, column:int, name:Optional[str]=None, health:int=-1, damage:int=-1, aoe:int=-1, food:int=100, weapon:str=None, *args, **kwargs):
        super().__init__(row, column, name, health, damage, aoe, food)
        Human.counter += 1
        self.id = Human.counter
        
        self.weapon = None
    

    def arm(self, weapon: Object):
        
        if self.weapon == None:
            self.weapon = weapon

            if self.weapon.attribute == "health":
                self.health += self.weapon.amount
            elif self.weapon.attribute == "damage":
                self.damage += self.weapon.amount
            elif self.weapon.attribute == "aoe":
                self.aoe += self.weapon.amount
            else:
                raise RuntimeError
    
    def own_weapon(self):
        if self.weapon != None:
            return str(self.weapon.info())
        else:
            return "None"
        
    def info(self):
        return (self.name if self.name else self.__class__.__name__) + \
        " {0.id}, row={0.row}, column={0.column}, health={0.health}, damage={0.damage}, aoe={0.aoe}, food={0.food}, ".format(self) + \
        self.own_weapon()


    def __del__(self):
        Human.counter -= 1


class Zombie(Character):
    counter = 0

    HEALTH = range(10, 50)
    DAMAGE = range(1, 3)
    AOE = range(1, 2)

    def __init__(self, row:int, column:int, name:Optional[str]=None, health:int=-1, damage:int=-1, aoe:int=-1, food:int=100, *args, **kwargs):
        super().__init__(row, column, name, health, damage, aoe, food)
        Zombie.counter += 1
        self.id = Zombie.counter
        
    def info(self):
        return (self.name if self.name else self.__class__.__name__) + \
        " {0.id}, row={0.row}, column={0.column}, health={0.health}, damage={0.damage}, aoe={0.aoe}, food={0.food}".format(self)


    def __del__(self):
        Zombie.counter -= 1


class Food(Object):
    counter = 0

    FOOD = range(10, 20)

    def __init__(self, row:int, column:int, name:Optional[str]=None, amount:int=-1, *args):
        super().__init__(row, column, name)
        Food.counter += 1
        self.id = Food.counter

        self.amount = random.choice(self.FOOD)

    def __del__(self):
        Food.counter -= 1

    def __str__(self):
        return self.__class__.__name__


class Weapon(Object):
    counter = 0

    ATTRIBUTE = {"health": range(10, 20), "damage": range(5, 10), "aoe": range(1, 2)}

    def __init__(self, row:int, column:int, name:Optional[str]=None, attribute:Optional[str]=None, amount:int=-1, *args):
        super().__init__(row, column, name)
        Weapon.counter += 1
        self.id = Weapon.counter
        
        self.attribute = random.choice(list(self.ATTRIBUTE.keys()))
        self.amount = random.choice(self.ATTRIBUTE[self.attribute])
        
    def info(self):
        return self.attribute, self.amount

    def __del__(self):
        Weapon.counter -= 1

    def __str__(self):
        return self.__class__.__name__


def test_characters():
    Human.counter = 0
    human1 = Human(0, 0)
    print(human1)
    print(human1.info())
    human1.hit(2)
    print(human1.info())
    human1.heal(3)
    print(human1.info())
    food1 = Food(0,1)
    print(food1.amount)
    human1.eat(food1)
    print(human1.info())
    weapon1 = Weapon(0,3)
    print(weapon1.info())
    human1.arm(weapon1)
    print(human1.weapon)
    print(human1.info())
    move_row, move_column = human1.direction()
    human1.move(human1.row+move_row, human1.column+move_column)
    print(move_row, move_column)
    print(human1.info())
    human2 = Human(0, 0)
    print(human2.info())
    print(Human.counter)
    zombie1 = Zombie(0, 1, damage=100, aoe=10)
    print(zombie1.info())



from numbers import Real
from typing import Optional

def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5

class Grid:
    def __init__(self, m: int, n: int, food_prob: float, weapon_prob: float):
        self.m, self.n = m, n
        self.grid: list[list[Optional[Object]]] = [[None] * n for _ in range(m)]
        self.food_prob = food_prob
        self.weapon_prob = weapon_prob

    def add(self, obj: Object):
        if self.grid[obj.row][obj.column] is None:
            self.grid[obj.row][obj.column] = obj
        else:
            raise RuntimeError
      
    def size(self):
        """Return the size of the grid"""
        return str(self.m) + " * " + str(self.n)

    def count(self):
        count = ""
        for objects in {Human, Zombie, Food, Weapon}:
            count += "\n" + str(objects.__name__) + ": " + str(objects.counter)
        none_number = self.m * self.n - Human.counter - Zombie.counter - Food.counter - Weapon.counter
        count += "\n" + str(None) + ": " + str(none_number)
        return count

    def attacked(self, character):
        blood = 0
        if isinstance(character, Zombie) == True:
            for enemy_row in range(self.m):
                for enemy_col in range(self.n):
                    enemy = self.grid[enemy_row][enemy_col]
                    if isinstance(enemy, Human):
                        if euclidean_distance([enemy_row, enemy_col], [character.row, character.column]) <= enemy.aoe:
                            blood += enemy.damage
            return blood
        if isinstance(character, Human) == True:
            for enemy_row in range(self.m):
                for enemy_col in range(self.n):
                    enemy = self.grid[enemy_row][enemy_col]
                    if isinstance(enemy, Zombie):
                        if euclidean_distance([enemy_row, enemy_col], [character.row, character.column]) <= enemy.aoe:
                            blood += enemy.damage
            return blood            

    def update(self):
        for row in range(self.m):
            for col in range(self.n):
                character = self.grid[row][col]
#                print("character first round", Food.counter)
                if (isinstance(character, Human) or isinstance(character, Zombie)):
                    destination = character
                    move_row = 0
                    move_column = 0
                    trials = 0
                    while (isinstance(destination, Human) or isinstance(destination, Zombie) or
                        (isinstance(character, Zombie) and isinstance(destination, Weapon))
                        or row+move_row < 0 or row+move_row > self.m-1
                        or col+move_column < 0 or col+move_column > self.n-1):
                        move_row, move_column = character.direction()
                        try:
                            destination = self.grid[row+move_row][col+move_column]
                        except:
                            continue
                        trials +=1
                        if trials >=1000:
                            move_row = 0
                            move_column = 0
                            destination = self.grid[row+move_row][col+move_column]
                            break
#                    print("destination found", Food.counter)
#                    print(row+move_row, col+move_column)
                    if isinstance(destination, Food):
                        character.eat(destination)
                        self.grid[row+move_row][col+move_column] = None
                        del destination
#                        print("food deleted")
                    elif (isinstance(character, Human) and isinstance(destination, Weapon)):
                        character.arm(destination)
                        self.grid[row+move_row][col+move_column] = None
                        del destination
#                        print("weapon deleted")
                    character.move(row+move_row, col+move_column) # not yet tested
#                    print("*character moved", Food.counter)
                    self.grid[row][col] = None
                    self.add(character)
#                    print("character added", Food.counter)
        for row in range(self.m):
            for col in range(self.n):
                character = self.grid[row][col]
#                print("character second round", Food.counter)           
                if (isinstance(character, Human) or isinstance(character, Zombie)):    
                    blood = self.attacked(character)
#                    print("character attacked", Food.counter)
                    character.hit(blood)
#                    print("*character hit", Food.counter)
                    character.starve()
#                    print("character starve", Food.counter)
                    if character.health <= 0:
                        self.grid[row][col] = None
                        del character
#                        print("character deleted", Food.counter)
                    else:
                        character.heal(1)
#                        print("character healed", Food.counter)
                else:                    
                    if character == None:
                        if random.random() > self.food_prob: # SIR model
                          food = Food(row, col)
                          self.add(food)
#                          print("food added", Food.counter)
                        elif random.random() > self.weapon_prob: # SIR model
                          weapon = Weapon(row, col)
                          self.add(weapon)
#                          print("food added", Food.counter)

    # new describe method after other method works
    def show(self):
        return '\n'.join('\t'.join(f'{str(o):3s}' if o else ' . 'for o in row)
            for row in self.grid)
 
    def describe(self):
        all_characters = ""
        for row in range(self.m):
            for col in range(self.n):
                character = self.grid[row][col]
                if (isinstance(character, Human) or isinstance(character, Zombie)):
                    all_characters += "\n" + character.info()
        return all_characters


def test_environment():
    grid = Grid(m=4, n=4, food_prob=0.001, weapon_prob=0.1)
    print(grid.size())

    Human.counter = 0
    Zombie.counter = 0
    Food.counter = 0
    Weapon.counter = 0

    while (Human.counter < 2):
        row = random.choice(range(0, grid.m))
        col = random.choice(range(0, grid.n))
        if (grid.grid[row][col] == None):      
            grid.add(Human(row, col, name='H'))

    while (Zombie.counter < 2):        
        row = random.choice(range(0, grid.m))
        col = random.choice(range(0, grid.n))
        if grid.grid[row][col] == None:
            grid.add(Zombie(row, col, name='Z'))

    while (Food.counter < 1):
        row = random.choice(range(0, grid.m))
        col = random.choice(range(0, grid.n))
        if grid.grid[row][col] == None:
            grid.add(Food(row, col))
    
    while (Weapon.counter < 5):
        row = random.choice(range(0, grid.m))
        col = random.choice(range(0, grid.n))
        if grid.grid[row][col] == None:
            grid.add(Weapon(row, col))
    """
    # not implemented yet
    # overlapped with add + runtime method
    def add_random(self, obj: Object):
        none_number = grid.m * grid.n - Human.counter - Zombie.counter - Food.counter
        if none_number > 0:
            choices = [i for i, x in enumerate(self.grid) if x==None]
            index = random.choice(choices)
            self.grid[index] = obj
    """

    print(grid.show())
    print(grid.describe())
    print(grid.count())

    for _ in range(50):
        grid.update()
        print()
        print(grid.show())
        print(grid.describe())
        print(grid.count())

test_characters()
#test_environment()



