"""
moved two boxes each time: assign direction and move ascending to the id (a list that may change to other order)
(consider the possibility that every action like movement, get weapon, physical training, make weapon, turn into zombie take time in the future)
show info according to index
1. based on own condition (random, born, previous condition)
2. based on environment (place condition, proximity of others)
human when health < 0, either eat human or turn human into zombies (in grid)
turn checking of if a move is legal or not to separate function in grid
spawn food = 100 is the max, spawn health is the max, make use of other spawn info
food and weapon and armour turn into resources that human can store
weapon replacement
function to control the spawn rate of food and weapon in different terrain to allow different playing experience
use normal distribution or the Poisson distribution for human and zombie attribute
win condition
what happens when the cloest human is right next to the agent
list comprehension
map, filter function
Dictionary for storage
Refactoring, extract function, early exit
no multiple inheritance
custom error class inherit from Exception, it can raise error to pass through useful attributes
setattr can put a attribute dictionary into a class using for loop (dynamically)
getattr can get a attribute dictionary from a class using for loop (dynamically)
https://www.programiz.com/python-programming/decorator
can use @classmethod to return counter which depends on class instead of instance,
so either class.return or instance.return will give counter of class
can use @staticmethod to convert human to zombie
so as to control the activity of the class using a outside variable, without depending on class and instance, 
so function related to class but does not interact with class and instance can be grouped under the class
use @abstractmethod for method that must be reimplemented in subclass and do not interact with instance and class
pathfinding to a specific spot in the determination of direction
dataclass to store struct-like data, like dictionary, vs behaviour data
teacher one class, student one class
change to float for faster performance
group of zombies and groups of humans
call method vs init method (__call__ can return an arbitrary value whereas __init__ must return None, init is called when instantiating the class: myfoo = Foo(1,4,7.8) call is a template to call the already instantiated class to do something, 
__call__ not only allows an instance to be used as a function, it defines the function body that is executed when an instance is used as a function.)
a = A()     init
a()        call
better way to store objects in python
https://realpython.com/inner-functions-what-are-they-good-for/
functional programming 教學
https://medium.com/onedegree-tech-blog/%E7%82%BA%E4%BB%80%E9%BA%BC%E6%88%91%E8%A6%81%E5%AD%B8-functional-programming-%E5%8F%AF%E8%AE%80-%E5%8F%AF%E9%9D%A0-%E6%9C%89%E4%BF%A1%E5%BF%83-160e39f1632b
https://www.youtube.com/watch?v=ZtInesLXD-Y
"""

# python decorator to count time it take for a function using time.time()
from typing import Optional
import random


def time_it(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Total time taken in : {func.__name__} is {end - start}")
        return result
    return wrapper

# use functools for debugging


def debug(func):
    import functools

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


# object class
class Object:
    def __init__(self, row: int, column: int, name: Optional[str] = None):

        # set row and column
        self.row, self.column = row, column

        # set name
        self.name = name

# food class


class Food(Object):
    counter = 0

    # init food
    FOOD = range(10, 20)

    def __init__(self, row: int, column: int, name: Optional[str] = None, amount: int = -1, *args):
        super().__init__(row, column, name)

        # count food
        Food.counter += 1
        self.id = Food.counter

        # set food
        self.amount = random.choice(self.FOOD)

    # delete
    def __del__(self):
        Food.counter -= 1

    # name
    def __str__(self):
        return self.__class__.__name__

# weapon class


class Weapon(Object):
    counter = 0

    # init attribute
    ATTRIBUTE = {"health": range(10, 20), "damage": range(
        5, 10), "aoe": range(1, 2)}

    def __init__(self, row: int, column: int, name: Optional[str] = None, attribute: Optional[str] = None, amount: int = -1, *args):
        super().__init__(row, column, name)
        # count weapons
        Weapon.counter += 1
        self.id = Weapon.counter

        # set attribute and amount
        self.attribute = random.choice(list(self.ATTRIBUTE.keys()))
        self.amount = random.choice(self.ATTRIBUTE[self.attribute])

    # info
    def info(self) -> str:
        return "{0.self.attribute}, amount= {0.self.amount}".format(self)

    # delete
    def __del__(self):
        Weapon.counter -= 1

    # name
    def __str__(self):
        return self.__class__.__name__

# character class


class Character(Object):

    # init health, damage, aoe
    HEALTH = range(0, 1)
    DAMAGE = range(0, 1)
    AOE = range(0, 1)

    def __init__(self, row: int, column: int, name: Optional[str] = None, health: int = -1, damage: int = -1, aoe: int = -1, food: int = 100, *args, **kwargs):
        super().__init__(row, column, name)

        # set health, damage, aoe, food
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

    # get direction
    def direction(self) -> tuple:
        move_row = 0
        move_column = 0
        move_speed = random.choice(range(-1, 2))
        if (random.random() > 0.5):
            move_row = move_speed
        else:
            move_column = move_speed
        return move_row, move_column

    # move
    def move(self, destination_row, destination_column) -> None:
        if self.row != destination_row:
            self.food -= 1
        if self.column != destination_column:
            self.food -= 1
        self.row = destination_row
        self.column = destination_column

    # attack
    def attack(self, damage: int) -> None:
        self.health -= damage

    # eat
    def eat(self, food: Food) -> None:
        self.food += food.amount

    # starve
    def starve(self) -> None:
        if self.food <= 0:
            self.health -= 1

    # heal
    def heal(self, amount: int) -> None:
        self.health = min(self.health + amount, self.spawn_health)

    # name
    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__class__.__name__

# human class


class Human(Character):
    counter = 0

    # init human health, damage, aoe
    HEALTH = range(10, 50)
    DAMAGE = range(1, 3)
    AOE = range(1, 2)

    # make use of character id and counter (faster calculation or import models)
    # separate count method

    def __init__(self, row: int, column: int, name: Optional[str] = None, health: int = -1, damage: int = -1, aoe: int = -1, food: int = 100, weapon: Optional[Weapon] = None, *args, **kwargs):

        # get human health, damage, aoe
        super().__init__(row, column, name, health, damage, aoe, food)

        # count humans
        Human.counter += 1
        self.id = Human.counter

        # init weapon
        self.weapon = weapon

    # arm
    def arm(self, weapon: Weapon) -> None:

        if self.weapon == None:
            self.weapon = weapon

            if self.weapon.attribute == "health":
                self.health += self.weapon.amount
            elif self.weapon.attribute == "damage":
                self.damage += self.weapon.amount
            elif self.weapon.attribute == "aoe":
                self.aoe += self.weapon.amount
            else:
                raise RuntimeError("Invalid weapon attribute")

    # weapon info
    def own_weapon(self) -> str:
        if self.weapon != None:
            return "weapon: " + self.weapon.info()
        else:
            return "None"

    # info
    def info(self) -> str:
        return (self.name if self.name else self.__class__.__name__) + \
            " {0.id}, row={0.row}, column={0.column}, health={0.health}, damage={0.damage}, aoe={0.aoe}, food={0.food}, ".format(self) + \
            self.own_weapon()

    # delete
    def __del__(self):
        Human.counter -= 1

# zombie class


class Zombie(Character):
    counter = 0

    # init zombie health, damage, aoe
    HEALTH = range(10, 50)
    DAMAGE = range(1, 3)
    AOE = range(1, 2)

    def __init__(self, row: int, column: int, name: Optional[str] = None, health: int = -1, damage: int = -1, aoe: int = -1, food: int = 100, *args, **kwargs):

        # get human health, damage, aoe
        super().__init__(row, column, name, health, damage, aoe, food)

        # count zombies
        Zombie.counter += 1
        self.id = Zombie.counter

    # info
    def info(self) -> str:
        return (self.name if self.name else self.__class__.__name__) + \
            " {0.id}, row={0.row}, column={0.column}, health={0.health}, damage={0.damage}, aoe={0.aoe}, food={0.food}".format(
                self)

    # delete
    def __del__(self):
        Zombie.counter -= 1


# test characters
def test_characters():
    Human.counter = 0
    human1 = Human(0, 0)
    print(human1)
    print(human1.info())
    human1.attack(2)
    print(human1.info())
    human1.heal(3)
    print(human1.info())
    food1 = Food(0, 1)
    print(food1.amount)
    human1.eat(food1)
    print(human1.info())
    weapon1 = Weapon(0, 3)
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


# distance function

def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5

# grid class


class Grid:
    def __init__(self, m: int, n: int, food_prob: float, weapon_prob: float):
        # set grid size
        self.m, self.n = m, n

        # set grid
        self.grid: list[list[Optional[Object]]] = [
            [None] * n for _ in range(m)]

        # set food and weapon probability
        self.food_prob = food_prob
        self.weapon_prob = weapon_prob

    # add object to grid
    def add(self, obj: Object) -> None:
        if self.grid[obj.row][obj.column] is None:
            self.grid[obj.row][obj.column] = obj
        else:
            raise RuntimeError("Cannot add object to occupied position")

    # grid size
    def size(self) -> str:
        """Return the size of the grid"""
        return str(self.m) + " * " + str(self.n)

    # count character
    def count(self) -> str:
        count = ""
        for objects in {Human, Zombie, Food, Weapon}:
            count += "\n" + str(objects.__name__) + ": " + str(objects.counter)
        none_number = self.m * self.n - Human.counter - \
            Zombie.counter - Food.counter - Weapon.counter
        count += "\n" + str(None) + ": " + str(none_number)
        return count

    # calculate damage received by a character
    def attacked(self, character) -> int:
        blood = 0
        if isinstance(character, Zombie) == True:
            for enemy_row in range(self.m):
                for enemy_col in range(self.n):
                    enemy = self.grid[enemy_row][enemy_col]
                    if isinstance(enemy, Human):
                        if euclidean_distance([enemy_row, enemy_col], [character.row, character.column]) <= enemy.aoe:
                            blood += enemy.damage
        elif isinstance(character, Human) == True:
            for enemy_row in range(self.m):
                for enemy_col in range(self.n):
                    enemy = self.grid[enemy_row][enemy_col]
                    if isinstance(enemy, Zombie):
                        if euclidean_distance([enemy_row, enemy_col], [character.row, character.column]) <= enemy.aoe:
                            blood += enemy.damage
        return blood

    # update grid
    def update(self) -> None:
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
                           (isinstance(character, Zombie)
                            and isinstance(destination, Weapon))
                           or row+move_row < 0 or row+move_row > self.m-1
                            or col+move_column < 0 or col+move_column > self.n-1):
                        move_row, move_column = character.direction()
                        try:
                            destination = self.grid[row +
                                                    move_row][col+move_column]
                        except:
                            continue
                        trials += 1
                        if trials >= 1000:
                            move_row = 0
                            move_column = 0
                            destination = self.grid[row +
                                                    move_row][col+move_column]
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
                    # not yet tested
                    character.move(row+move_row, col+move_column)
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
                    character.attack(blood)
                    print("*character attack", Food.counter)
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
                        if random.random() < self.food_prob:  # SIR model
                            food = Food(row, col)
                            self.add(food)
#                          print("food added", Food.counter)
                        elif random.random() < self.weapon_prob:  # SIR model
                            weapon = Weapon(row, col)
                            self.add(weapon)
#                          print("food added", Food.counter)

    # new describe method after other method works

    # print grid
    def show(self) -> str:
        return '\n'.join('\t'.join(f'{str(o):3s}' if o else ' . 'for o in row)
                         for row in self.grid)

    # describe grid
    def describe(self) -> str:
        all_characters = ""
        for row in range(self.m):
            for col in range(self.n):
                character = self.grid[row][col]
                if (isinstance(character, Human) or isinstance(character, Zombie)):
                    all_characters += "\n" + character.info()
        return all_characters

# test environment


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
        else:
            raise RuntimeError("No more space to add object")
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

# test_characters()
# test_environment()
