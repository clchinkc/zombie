import random


class Survivor:
    def __init__(self, name, strength):
        self.name = name
        self.strength = strength
        self.is_alive = True

    def attack(self):
        attack_strength = random.randint(1, self.strength)
        return attack_strength

    def take_damage(self, damage):
        self.strength -= damage
        if self.strength <= 0:
            self.is_alive = False

class Horde:
    def __init__(self, name, num_zombies):
        self.name = name
        self.zombies = []
        for i in range(num_zombies):
            zombie = Survivor(f"{self.name}_zombie{i+1}", random.randint(1, 10))
            self.zombies.append(zombie)

    def is_defeated(self):
        return all(not zombie.is_alive for zombie in self.zombies)

    def select_random_zombie(self):
        alive_zombies = [zombie for zombie in self.zombies if zombie.is_alive]
        return random.choice(alive_zombies)

def simulate_apocalypse(survivor_group, zombie_horde):
    print("Zombie apocalypse simulation begins!")
    round_count = 1

    while not survivor_group.is_defeated() and not zombie_horde.is_defeated():
        print(f"\n--- Round {round_count} ---")
        survivor = survivor_group.select_random_zombie()
        zombie = zombie_horde.select_random_zombie()
        print(f"{survivor.name} attacks {zombie.name}.")

        attack_strength = survivor.attack()
        zombie.take_damage(attack_strength)
        print(f"{zombie.name} takes {attack_strength} damage.")

        if not zombie.is_alive:
            print(f"{zombie.name} has been defeated!")

        survivor_group_members_remaining = sum(survivor.is_alive for survivor in survivor_group.zombies)
        zombie_horde_members_remaining = sum(zombie.is_alive for zombie in zombie_horde.zombies)
        print(f"{survivor_group.name} has {survivor_group_members_remaining} members remaining.")
        print(f"{zombie_horde.name} has {zombie_horde_members_remaining} zombies remaining.")

        round_count += 1

    if survivor_group.is_defeated() and zombie_horde.is_defeated():
        print("\nThe apocalypse ended in a draw!")
    elif survivor_group.is_defeated():
        print(f"\n{zombie_horde.name} has overwhelmed the survivors!")
    else:
        print(f"\n{survivor_group.name} survived the apocalypse!")

# Example usage:
survivor_group = Horde("Survivor Group", 5)
zombie_horde = Horde("Zombie Horde", 5)
simulate_apocalypse(survivor_group, zombie_horde)


"""
Designing a complete zombie apocalypse simulation code can be a complex task, but I can provide you with a general framework and considerations to get you started. Please note that the following design is a simplified version and may require further refinement based on your specific requirements and programming language of choice.

1. Game Setup:
   - Define the groups involved: Create a list of groups, such as Survivor Group A, Survivor Group B, and so on. Each group will have its own characteristics, strengths, weaknesses, and members.
   - Define the urban environment: Set up the urban setting with terrain features, such as buildings, streets, parks, and hideouts. Each terrain may affect movement, visibility, and combat outcomes.
   - Assign members: Each group should have a set of members with different attributes like health, attack strength, defense, agility, and range.

2. Turn-Based Mechanics:
   - Initiative: Determine the order in which groups take turns based on a predetermined initiative system. This can be randomized or determined by specific factors like leadership skills.
   - Actions per turn: Decide the number of actions each member can take per turn. This can include movements, attacks, or other special abilities.

3. Movement:
   - Member positioning: Establish a grid-based system or any other appropriate mechanism to position members in the urban environment.
   - Movement rules: Define movement mechanics based on member attributes and urban terrain features. For example, agile members may have increased movement range, while heavily armed members may be slower but more resilient.

4. Combat Mechanics:
   - Target selection: Determine how members select their targets based on proximity, threat level, or strategic considerations.
   - Attack and defense: Create rules for calculating the outcome of an attack based on factors like attack strength, defense, member type, and any special abilities or modifiers.
   - Damage calculation: Determine the damage inflicted on the defending member or zombie based on the attack outcome and defensive attributes.

5. Morale and Leadership:
   - Morale system: Implement a morale system to simulate the psychological state of members. Events like casualties, leadership presence, or unexpected zombie hordes can affect morale.
   - Leadership effects: Include the influence of group leaders or key figures on members, such as morale boosts or special abilities.

6. Survival Conditions:
   - Define survival conditions for each group, such as securing a stronghold, gathering a certain amount of resources, or saving a specific number of survivors.

7. User Interface:
   - Develop a user interface to display the urban setting, member information, and relevant game statistics. Allow players to issue commands and observe the ongoing apocalypse.

8. AI Zombies (optional):
   - Implement an artificial intelligence system to control zombie hordes not controlled by players. The AI should make strategic decisions, control zombie movements, and engage in combat.

Remember, this is just a basic framework for a zombie apocalypse simulation. You can further expand and refine the mechanics based on your desired complexity and the specific features you want to incorporate into your simulation.
"""
