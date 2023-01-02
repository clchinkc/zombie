import math
import random
import numpy as np

class ZombieApocalypse:
    def __init__(self, num_zombies, num_survivors):
        self.num_zombies = num_zombies
        self.num_survivors = num_survivors
        self.survivors = []
        self.zombies = []

    def simulate(self):
        # Initialize survivors and zombies
        for i in range(self.num_survivors):
            self.survivors.append(Survivor(i))
        for i in range(self.num_zombies):
            self.zombies.append(Zombie(i))
            
        # list possible weapons
        possible_weapons = [
            Weapon("Baseball Bat", 20, 2),
            Weapon("Pistol", 30, 5),
            Weapon("Rifle", 40, 8),
            Weapon("Molotov Cocktail", 50, 3),
        ]

        # simulate activities of survivors and zombies
        while self.num_survivors > 0 and self.num_zombies > 0:
            for survivor in self.survivors:
                survivor.take_turn(possible_weapons)
            for zombie in self.zombies:
                zombie.take_turn()

        # Check if survivors or zombies won
        if self.num_survivors == 0:
            print("Zombies have won!")
        else:
            print("Survivors have won!")
            
    # as escape location or method

class Human:
    def __init__(self, id):
        self.id = id
        self.health = 100
        self.location = (0, 0) # x, y coordinates on the map
        
        # health, strength, speed
        # inherit from individual class
        # speed controls who attacks first, if dead can't attack back
        # or not in turn-based game, attack in interval of speed time

    def distance(self, other):
        # Calculate the distance between two humans using the Pythagorean theorem
        x_distance = abs(self.location[0] - other.location[0])
        y_distance = abs(self.location[1] - other.location[1])
        distance = int(math.sqrt(x_distance ** 2 + y_distance ** 2))
        return distance
            
    def get_enemies_in_range(self):
        # Calculate the distance to each enemy
        distances = []
        if isinstance(self, Zombie):
            enemies_list = apocalypse.survivors
        else:
            enemies_list = apocalypse.zombies
        for enemy in enemies_list:
            distance = self.distance(enemy)
            distances.append((distance, enemy))

        # Sort the list of enemies by distance
        distances.sort(key=lambda x: x[0])

        # Return a list of enemies within range
        enemies_in_range = []
        attack_range = math.sqrt(2+ self.weapon.range) if (isinstance(self, Survivor) and self.weapon is not None) else math.sqrt(2)
        for distance, enemy in distances:
            if distance <= attack_range:
                enemies_in_range.append(enemy)
            else:
                break

        return enemies_in_range
    
    def get_closest_enemies(self, enemies_list):
        # Calculate the distance to each enemy
        distances = []
        for enemy in enemies_list:
            distance = self.distance(enemy)
            distances.append((distance, enemy))

        # Sort the list of enemies by distance
        distances.sort(key=lambda x: x[0])

        # Return the closest enemy
        return distances[0][1]
    
    def move(self, dx, dy):
        # Check if the chosen location is already occupied
        occupied = False
        for survivor in apocalypse.survivors:
            if survivor.location == (self.location[0]+dx, self.location[1]+dy):
                occupied = True
                break
        for zombie in apocalypse.zombies:
            if zombie.location == (self.location[0]+dx, self.location[1]+dy):
                occupied = True
                break

        # If the chosen location is not occupied, update the survivor's location
        if not occupied:
            self.location = (self.location[0]+dx, self.location[1]+dy)

    def take_damage(self, damage):
        # Reduce the health of the human by the specified amount
        self.health -= damage

    # strength and defense attributes
    # defend method to reduce damage taken
    
    """
    def attack(self, target):
        # if the target is a zombie, fight
            # if the human is faster than the zombie, they hit first
                # the zombie's health is reduced by the human's strength
                # if the zombie is still alive, attack back
                    # the human's health is reduced by the zombie's strength
                    # if the human is dead, turn to zombie
                # if the zombie is dead, die
            # if the zombie is faster than the human, they hit first
                # the human's health is reduced by the zombie's strength
                # if the human is still alive, attack back
                    # the zombie's health is reduced by the human's strength
                    # if the zombie is dead, die
                # if the human is dead, turn to zombie
    """

class Survivor(Human):
    def __init__(self, id):
        super().__init__(id)
        self.weapon = None

    def take_turn(self, possible_weapons):
        # Check if there are any zombies in range
        zombies_in_range = self.get_enemies_in_range()
        if len(zombies_in_range) > 0:
            # Attack the closest zombie
            zombie_to_attack = self.get_closest_enemies(zombies_in_range)
            self.attack(zombie_to_attack)
        else:
            # No zombies in range, so scavenge for supplies or move to a new location
            if self.should_scavenge(possible_weapons):
                self.scavenge(possible_weapons)
            else:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                self.move(self.location[0]+dx, self.location[1]+dy)

    def should_scavenge(self, possible_weapons):
        # Scavenge if the survivor's health is below a certain threshold
        if self.health < 50:
            return True
        
        # Scavenge if the survivor doesn't have a weapon
        if self.weapon is None:
            return True
        
        # Scavenge if the survivor can find a better weapon
        for weapon in possible_weapons:
            if weapon.damage > self.weapon.damage:
                return True
        return False

    def scavenge(self, possible_weapons):
        # Roll a dice to determine if the survivor finds any supplies
        if random.random() < 0.5:
            # Survivor has found some supplies
            supplies = random.randint(10, 20)
            self.health += supplies
            print(f"Survivor {self.id} found {supplies} supplies!")

        # Roll a dice to determine if the survivor finds a new weapon
        if random.random() < 0.2:
            # Survivor has found a new weapon
            new_weapon = self.get_random_weapon(possible_weapons)
            if self.weapon is None or (new_weapon.damage * new_weapon.range > self.weapon.damage * self.weapon.range):
                self.weapon = new_weapon
                print(f"Survivor {self.id} found a new {new_weapon.name}!")


    def get_random_weapon(self, possible_weapons):
        # Choose a random weapon from a list of possible weapons
        return random.choice(possible_weapons)    
    
    def attack(self, zombie):
        # Calculate the damage dealt to the zombie
        damage = 10 if self.weapon == None else 10+self.weapon.damage

        # Deal the damage to the zombie
        zombie.take_damage(damage)
        print(f"Survivor {self.id} attacked zombie {zombie.id} for {damage} damage!")
            
    def take_damage(self, damage):
        # Reduce the survivor's health by the specified amount of damage
        super().take_damage(damage)
        print(f"Survivor {self.id} taking {damage} damage, new health is {self.health}")

        # Check if the survivor is dead
        if self.health <= 0:
            print(f"Survivor {self.id} has died!")
            # Remove the survivor from the list of survivors            
            apocalypse.survivors.remove(self)
            apocalypse.num_survivors -= 1
            # Create a new zombie at the survivor's location
            zombie = Zombie(apocalypse.num_zombies)
            apocalypse.zombies.append(zombie)
            zombie.location = self.location
            apocalypse.num_zombies += 1

class Zombie(Human):
    def __init__(self, id):
        super().__init__(id)

    def take_turn(self):
        # Check if there are any survivors in range
        survivors_in_range = self.get_enemies_in_range()
        if len(survivors_in_range) > 0:
            # Attack the closest survivor
            survivor_to_attack = self.get_closest_enemies(survivors_in_range)
            self.attack(survivor_to_attack)
        else:
            # No survivors  in range
            survivor_to_attack = self.get_closest_enemies(apocalypse.survivors)
            self.move_towards_survivor(survivor_to_attack)

    def attack(self, survivor):
        # Deal 10 damage to the survivor
        survivor.take_damage(20)
        print(f"Zombie {self.id} attacking survivor {survivor.id}")

    def take_damage(self, damage):
        super().take_damage(damage)
        print(f"Zombie {self.id} taking {damage} damage, new health is {self.health}")
        # Check if the zombie has been killed
        if self.health <= 0:
            # Remove the zombie from the list of zombies
            apocalypse.zombies.remove(self)
            apocalypse.num_zombies -= 1        
    
    def move_towards_survivor(self, survivor):
        # Calculate the direction in which the survivor is located
        x_diff = survivor.location[0] - self.location[0]
        y_diff = survivor.location[1] - self.location[1]

        # Move the zombie towards the survivor
        if x_diff > 0:
            self.move(1, 0)
        elif x_diff < 0:
            self.move(-1, 0)
        if y_diff > 0:
            self.move(0, 1)
        elif y_diff < 0:
            self.move(0, -1)

class Weapon:
    def __init__(self, name, damage, range):
        self.name = name
        self.damage = damage
        self.range = range
        
    def __str__(self):
        return f"{self.name} ({self.damage} damage, {self.range} range)"

# Create an instance of the Zombie Apocalypse simulation
#apocalypse = ZombieApocalypse(5, 5)

# Run the simulation
#apocalypse.simulate()
