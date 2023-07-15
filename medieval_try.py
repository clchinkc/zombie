import random


class Soldier:
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

class Army:
    def __init__(self, name, num_soldiers):
        self.name = name
        self.soldiers = []
        for i in range(num_soldiers):
            soldier = Soldier(f"{self.name}_soldier{i+1}", random.randint(1, 10))
            self.soldiers.append(soldier)

    def is_defeated(self):
        return all(not soldier.is_alive for soldier in self.soldiers)

    def select_random_soldier(self):
        alive_soldiers = [soldier for soldier in self.soldiers if soldier.is_alive]
        return random.choice(alive_soldiers)

def simulate_war(army1, army2):
    print("War simulation begins!")
    round_count = 1

    while not army1.is_defeated() and not army2.is_defeated():
        print(f"\n--- Round {round_count} ---")
        attacker = army1.select_random_soldier()
        defender = army2.select_random_soldier()
        print(f"{attacker.name} attacks {defender.name}.")

        attack_strength = attacker.attack()
        defender.take_damage(attack_strength)
        print(f"{defender.name} takes {attack_strength} damage.")

        if not defender.is_alive:
            print(f"{defender.name} has been defeated!")

        army1_soldiers_remaining = sum(soldier.is_alive for soldier in army1.soldiers)
        army2_soldiers_remaining = sum(soldier.is_alive for soldier in army2.soldiers)
        print(f"{army1.name} has {army1_soldiers_remaining} soldiers remaining.")
        print(f"{army2.name} has {army2_soldiers_remaining} soldiers remaining.")

        round_count += 1

    if army1.is_defeated() and army2.is_defeated():
        print("\nThe war ended in a draw!")
    elif army1.is_defeated():
        print(f"\n{army2.name} won the war!")
    else:
        print(f"\n{army1.name} won the war!")

# Example usage:
army1 = Army("Blue Army", 5)
army2 = Army("Red Army", 5)
simulate_war(army1, army2)

"""
Designing a complete medieval war simulation code can be a complex task, but I can provide you with a general framework and considerations to get you started. Please note that the following design is a simplified version and may require further refinement based on your specific requirements and programming language of choice.

1. Game Setup:
   - Define the factions involved: Create a list of factions, such as Kingdom A, Kingdom B, and so on. Each faction will have its own characteristics, strengths, weaknesses, and units.
   - Define the battlefield: Set up the battlefield with terrain features, such as forests, rivers, hills, and castles. Each terrain may affect movement, visibility, and combat outcomes.
   - Assign units: Each faction should have a set of units with different attributes like health, attack strength, defense, speed, and range.

2. Turn-Based Mechanics:
   - Initiative: Determine the order in which factions take turns based on a predetermined initiative system. This can be randomized or determined by specific factors like leadership skills.
   - Actions per turn: Decide the number of actions each unit can take per turn. This can include movements, attacks, or other special abilities.

3. Movement:
   - Unit positioning: Establish a grid-based system or any other appropriate mechanism to position units on the battlefield.
   - Movement rules: Define movement mechanics based on unit attributes and terrain features. For example, cavalry units may have increased movement range, while infantry may be slower but more resilient.

4. Combat Mechanics:
   - Target selection: Determine how units select their targets based on proximity, threat level, or strategic considerations.
   - Attack and defense: Create rules for calculating the outcome of an attack based on factors like attack strength, defense, unit type, and any special abilities or modifiers.
   - Damage calculation: Determine the damage inflicted on the defending unit based on the attack outcome and defensive attributes.

5. Morale and Leadership:
   - Morale system: Implement a morale system to simulate the psychological state of units. Events like casualties, leadership presence, or flanking maneuvers can affect morale.
   - Leadership effects: Include the influence of faction leaders or high-ranking officers on units, such as morale boosts or special abilities.

6. Victory Conditions:
   - Define victory conditions for each faction, such as capturing the enemy's stronghold, eliminating a certain number of enemy units, or occupying key strategic positions.

7. User Interface:
   - Develop a user interface to display the battlefield, unit information, and relevant game statistics. Allow players to issue commands and observe the ongoing battle.

8. AI Opponent (optional):
   - Implement an artificial intelligence system to control factions not controlled by players. The AI should make strategic decisions, control unit movements, and engage in combat.

Remember, this is just a basic framework for a medieval war simulation. You can further expand and refine the mechanics based on your desired complexity and the specific features you want to incorporate into your simulation.
"""
