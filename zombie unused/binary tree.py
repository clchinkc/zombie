
import random
import tkinter as tk
from tkinter import messagebox, simpledialog


class Survivor:
    def __init__(self, name, location, health_status, hunger_level=0, thirst_level=0):
        self.name = name
        self.location = location
        self.health_status = health_status
        self.hunger_level = hunger_level
        self.thirst_level = thirst_level
        self.left = None
        self.right = None

    def communicate(self, message):
        print(f"{self.name} communicates: {message}")

    def form_alliance(self, other_survivor):
        print(f"{self.name} forms an alliance with {other_survivor.name}")

    def attack_zombie(self, zombie):
        print(f"{self.name} attacks a zombie!")
        zombie.defend()

    def defend(self):
        print(f"{self.name} defends against a zombie!")

    def handle_random_event(self):
        event = random.randint(1, 3)  
        if event == 1:
            self.encounter_zombie()
        elif event == 2:
            self.find_resources()
        else:
            self.rest()

    def find_resources(self):
        print(f"{self.name} finds valuable resources!")

    def rest(self):
        print(f"{self.name} takes a rest.")

    def perform_task(self, task):
        print(f"{self.name} is performing {task}.")

    def heal(self):
        print(f"{self.name} is healing.")

    def gather_resources(self, resource, amount):
        print(f"{self.name} gathered {amount} {resource}.")

    def trade_resources(self, other_survivor, resource, amount):
        print(f"{self.name} trades {amount} {resource} with {other_survivor.name}.")

    def encounter_zombie(self):
        print(f"{self.name} encounters a zombie!")
        if random.random() < 0.3:
            self.health_status = "Infected"
            print(f"{self.name} got infected!")

    def update_health_status(self):
        if self.health_status == "Infected":
            self.health_status = "Worsening" 
        elif self.hunger_level >= 5:
            self.health_status = "Starving"


class Zombie:
    def __init__(self, location, infection_level=0, strength=10, speed=1):
        self.location = location
        self.infection_level = infection_level
        self.strength = strength
        self.speed = speed

    def detect_survivor(self, survivor):
        print(f"A zombie detects {survivor.name} at {survivor.location}")

    def chase(self, survivor):
        print(f"A zombie is chasing {survivor.name}")

    def attack_survivor(self, survivor):
        print(f"A zombie attacks {survivor.name}")
        survivor.defend()

    def defend(self):
        print("The zombie defends itself")


class SurvivorNetwork:
    def __init__(self):
        self.root = None

    def add_survivor(self, survivor, parent=None, is_left_child=False):
        if parent is None:
            self.root = survivor
        else:
            if is_left_child:
                parent.left = survivor
            else:
                parent.right = survivor

    def find_survivor(self, name, start_node=None):
        if start_node is None:
            start_node = self.root
        if start_node is None:
            return None
        if start_node.name == name:
            return start_node
        survivor = self.find_survivor(name, start_node.left)
        if survivor is not None:
            return survivor
        survivor = self.find_survivor(name, start_node.right)
        if survivor is not None:
            return survivor
        return None

    def update_survivor_location(self, name, new_location):
        survivor = self.find_survivor(name)
        if survivor is not None:
            survivor.location = new_location
            print(f"{name} moved to {new_location}")
        else:
            print(f"Survivor {name} not found.")

    def handle_encounter(self, survivor_name, zombie):
        survivor = self.find_survivor(survivor_name)
        if survivor is not None:
            survivor.encounter_zombie(zombie)
        else:
            print(f"Survivor {survivor_name} not found.")

    def remove_survivor(self, survivor):
        # Remove a survivor from the survivor network (binary tree)
        self.root = self.remove_survivor_recursive(self.root, survivor)

    def remove_survivor_recursive(self, current_node, survivor):
        # Recursively remove a survivor from the binary tree

        if current_node is None:
            return None

        if current_node == survivor:
            if current_node.left is None:
                return current_node.right
            if current_node.right is None:
                return current_node.left
            min_node = self.find_min_node(current_node.right)
            min_node.right = self.remove_min_node(current_node.right)
            min_node.left = current_node.left
            return min_node

        if survivor.name < current_node.name:
            current_node.left = self.remove_survivor_recursive(current_node.left, survivor)
        else:
            current_node.right = self.remove_survivor_recursive(current_node.right, survivor)

        return current_node

    def find_min_node(self, current_node):
        # Find the minimum node in a binary tree
        while current_node.left is not None:
            current_node = current_node.left
        return current_node
    
    def remove_min_node(self, current_node):
        # Remove the minimum node from a binary tree
        if current_node.left is None:
            return current_node.right
        current_node.left = self.remove_min_node(current_node.left)
        return current_node

    # Add other methods for managing survivor resources, relationships, etc.


class SurvivorSimulation:
    def __init__(self):
        self.survivor_network = SurvivorNetwork()
        self.resources = {
            "food": 100,
            "water": 100,
            "medical_supplies": 50
        }
        self.game_over = False
        self.score = 0

    def run_simulation(self):
        time_elapsed = 0

        while time_elapsed < 24 and not self.game_over:
            print(f"----- Hour {time_elapsed + 1} -----")
            self.update_survivors()
            self.handle_encounters()
            self.handle_random_events()
            self.perform_tasks()
            self.consume_resources()
            self.check_survivors_status()

            time_elapsed += 1

        if self.game_over:
            print("Game Over")
            print(f"Final Score: {self.score}")

    def update_survivors(self):
        # Update survivor attributes based on their actions, movement, etc.
        self.update_survivor_attributes(self.survivor_network.root)

    def update_survivor_attributes(self, survivor):
        # Recursively update survivor attributes in the binary tree

        if survivor is None:
            return

        survivor.update_health_status()
        if survivor.health_status == "Worsening":
            self.handle_survivor_death(survivor)
        elif survivor.health_status == "Infected":
            self.score -= 5

        self.update_survivor_attributes(survivor.left)
        self.update_survivor_attributes(survivor.right)

    def handle_encounters(self):
        # Handle survivor-zombie encounters
        self.handle_survivor_encounters(self.survivor_network.root)

    def handle_survivor_encounters(self, survivor):
        # Recursively handle encounters between survivors and zombies in the binary tree

        if survivor is None:
            return
        
        if survivor.health_status == "Infected":
            self.score -= 5

        zombies = self.detect_nearby_zombies(survivor)
        if zombies:
            for zombie in zombies:
                survivor.attack_zombie(zombie)
                zombie.attack_survivor(survivor)

        self.handle_survivor_encounters(survivor.left)
        self.handle_survivor_encounters(survivor.right)

    def detect_nearby_zombies(self, survivor):
        # Detect nearby zombies based on survivor location
        # Return a list of zombies within a certain range

        zombies = []  # Placeholder for zombie detection logic

        return zombies

    def handle_random_events(self):
        # Trigger random events for survivors
        self.trigger_random_events(self.survivor_network.root)

    def trigger_random_events(self, survivor):
        # Recursively trigger random events for survivors in the binary tree

        if survivor is None:
            return
        
        if survivor.health_status == "Infected":
            self.score -= 5

        survivor.handle_random_event()

        self.trigger_random_events(survivor.left)
        self.trigger_random_events(survivor.right)

    def perform_tasks(self):
        # Assign tasks to survivors and manage their progress and stamina
        self.assign_tasks(self.survivor_network.root)

    def assign_tasks(self, survivor):
        # Recursively assign tasks to survivors in the binary tree

        if survivor is None:
            return

        task = self.choose_task()
        survivor.perform_task(task)
        
        if task == "gathering resources":
            self.gather_resources(survivor)
        elif task == "trading resources":
            self.trade_resources(survivor)

        self.assign_tasks(survivor.left)
        self.assign_tasks(survivor.right)

    def choose_task(self):
        # Choose a task based on some logic (e.g., random selection, priority-based, etc.)
        tasks = ["scavenging", "fortifying the base", "searching for other survivors"]
        return random.choice(tasks)

    def consume_resources(self):
        # Calculate resource consumption and update resource levels accordingly
        self.consume_survivor_resources(self.survivor_network.root)

    def consume_survivor_resources(self, survivor):
        # Recursively consume resources for survivors in the binary tree

        if survivor is None:
            return
        
        if survivor.health_status == "Infected":
            self.score -= 2

        self.consume_resource(survivor, "food", 2)
        self.consume_resource(survivor, "water", 1)

        self.consume_survivor_resources(survivor.left)
        self.consume_survivor_resources(survivor.right)

    def consume_resource(self, survivor, resource, amount):
        # Consume a specific amount of a resource for a survivor

        if resource in self.resources:
            if self.resources[resource] >= amount:
                self.resources[resource] -= amount
                print(f"{survivor.name} consumed {amount} {resource}.")
            else:
                print(f"{survivor.name} could not consume {amount} {resource}. Insufficient resources.")
        else:
            print(f"Invalid resource: {resource}")

    def heal_survivors(self):
        # Heal injured survivors based on available medical supplies
        for survivor in self.get_injured_survivors():
            if self.resources["medical_supplies"] > 0:
                survivor.heal()
                self.resources["medical_supplies"] -= 1
                print(f"{survivor.name} has been healed.")
            else:
                print(f"Insufficient medical supplies to heal {survivor.name}.")

    def get_injured_survivors(self):
        # Get a list of injured survivors based on their health status
        injured_survivors = []
        self.collect_injured_survivors(self.survivor_network.root, injured_survivors)
        return injured_survivors

    def collect_injured_survivors(self, survivor, injured_survivors):
        # Recursively collect injured survivors in the binary tree

        if survivor is None:
            return

        if survivor.health_status == "Injured":
            injured_survivors.append(survivor)

        self.collect_injured_survivors(survivor.left, injured_survivors)
        self.collect_injured_survivors(survivor.right, injured_survivors)

    def gather_resources(self, survivor):
        # Allow survivors to gather resources based on their location
        if survivor.location == "Forest":
            survivor.gather_resources("wood", 5)
        elif survivor.location == "Lake":
            survivor.gather_resources("water", 3)

    def trade_resources(self, survivor):
        # Allow survivors to trade resources with other survivors
        other_survivor = self.get_random_survivor(survivor)
        if other_survivor:
            resource = random.choice(list(self.resources.keys()))
            amount = random.randint(1, 5)
            survivor.trade_resources(other_survivor, resource, amount)
            self.resources[resource] += amount
        else:
            print(f"No other survivor available for trading with {survivor.name}.")

    def get_random_survivor(self, survivor):
        # Get a random survivor for trading with the given survivor
        available_survivors = self.get_available_survivors(survivor)
        if available_survivors:
            return random.choice(available_survivors)
        return None

    def get_available_survivors(self, survivor):
        # Get a list of available survivors for trading
        available_survivors = []
        self.collect_available_survivors(self.survivor_network.root, survivor, available_survivors)
        available_survivors.remove(survivor)  # Remove the current survivor from the list
        return available_survivors

    def collect_available_survivors(self, current_survivor, survivor, available_survivors):
        # Recursively collect available survivors in the binary tree for trading

        if current_survivor is None:
            return

        if current_survivor != survivor:
            available_survivors.append(current_survivor)

        self.collect_available_survivors(current_survivor.left, survivor, available_survivors)
        self.collect_available_survivors(current_survivor.right, survivor, available_survivors)

    def handle_survivor_death(self, survivor):
        # Remove dead survivor from the survivor network and update resources, etc.
        self.survivor_network.remove_survivor(survivor)
        print(f"{survivor.name} has died.")
        
        if survivor.health_status == "Infected":
            self.score -= 10
        elif survivor.health_status == "Starving":
            self.score -= 5
        # Handle other consequences of survivor death

    def check_survivors_status(self):
        # Checking if all survivors are either dead or infected
        survivor_status = self.get_survivors_status(self.survivor_network.root)

        if not any(status == "Healthy" for status in survivor_status):
            self.game_over = True

    def get_survivors_status(self, survivor):
        # Recursively get status of all survivors in the binary tree
        if survivor is None:
            return []

        survivor_status = [survivor.health_status]

        survivor_status.extend(self.get_survivors_status(survivor.left))
        survivor_status.extend(self.get_survivors_status(survivor.right))

        return survivor_status


class SurvivorGUI:
    def __init__(self, simulation):
        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("Survivor Game")

        # Create a canvas to display the game
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='white')
        self.canvas.pack()

        # Add a button to start the game
        self.start_button = tk.Button(self.root, text="Start Game", command=self.start_game)
        self.start_button.pack()

        # Add a button to end the game
        self.end_button = tk.Button(self.root, text="End Game", command=self.end_game)
        self.end_button.pack()

        # Add a button to add a survivor
        self.add_survivor_button = tk.Button(self.root, text="Add Survivor", command=self.add_survivor)
        self.add_survivor_button.pack()

        # Add a button to remove a survivor
        self.remove_survivor_button = tk.Button(self.root, text="Remove Survivor", command=self.remove_survivor)
        self.remove_survivor_button.pack()

    def start_game(self):
        # Start the simulation and update the display
        self.simulation.run_simulation()
        self.update_display()

    def add_survivor(self):
        # Add a survivor to the simulation
        name = simpledialog.askstring("Add Survivor", "Enter the name of the survivor")
        location = random.choice(["Forest", "Lake"])
        health_status = simpledialog.askstring("Add Survivor", "Enter the health status of the survivor (Healthy/ Injured/ Infected/ Starving)")
        self.simulation.survivor_network.add_survivor(Survivor(name, location, health_status))
        self.show_message(f"{name} added to the survivor network.")

    def remove_survivor(self):
        survivor = self.simulation.survivor_network.root
        if survivor:
            self.simulation.survivor_network.remove_survivor(survivor)
            self.show_message(f"{survivor.name} removed from the survivor network.")
        else:
            self.show_message("No survivor available for removal.")

    def end_game(self):
        # End the simulation and clear the display
        self.simulation.game_over = True
        self.canvas.delete("all")

    def update_display(self):
        # Update the display based on the current state of the simulation
        self.canvas.delete("all")
        self.draw_survivor_network(self.simulation.survivor_network.root, 400, 50, 400)
        
    def draw_survivor_network(self, survivor, x, y, x_shift):
        # Recursively draw the survivor network
        if survivor is None:
            return

        self.draw_survivor(survivor, x, y)
        self.draw_survivor_network(survivor.left, x - x_shift, y + 50, x_shift / 2)
        self.draw_survivor_network(survivor.right, x + x_shift, y + 50, x_shift / 2)
        
    def draw_survivor(self, survivor, x, y):
        # Draw a survivor
        self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30, fill="red")
        self.canvas.create_text(x, y, text=survivor.name)
        self.canvas.create_text(x, y + 20, text=survivor.health_status)

    def show_message(self, message):
        # Show a message to the user
        messagebox.showinfo("Message", message)

    def run(self):
        # Start the GUI
        self.root.mainloop()


if __name__ == "__main__":
    # Create a simulation object and run the GUI
    simulation = SurvivorSimulation()
    gui = SurvivorGUI(simulation)
    gui.run()

"""
Visualization and User Interface: Enhance the code with a graphical or text-based user interface to visualize the simulation and provide user interaction. Display survivor and zombie locations on a map, show their attributes and states, and provide options for the user to issue commands or make decisions for the survivors. Visual representation and user interface make the simulation more immersive and engaging.

You can further customize the simulation loop by adding additional functionality, such as survivor actions based on user input, displaying the current state of survivors and resources, introducing decision-making logic, etc.
"""

"""
To create a more realistic zombie apocalypse simulation, you would typically need to use a game development framework or engine that provides graphical capabilities. Rewriting the entire simulation code for a graphical game is beyond the scope of a text-based response, but I can provide you with a general overview of how you can approach building a realer zombie apocalypse simulation.

1. Game Development Framework: Choose a game development framework or engine that suits your needs. Popular options include Unity, Unreal Engine, and Godot. These frameworks provide the necessary tools and APIs to build interactive games with graphics, physics, and animations.

2. Game Objects and Assets: Design or acquire assets for your game, including 3D models for survivors, zombies, environments, and other game objects. You can find pre-made assets in online marketplaces or create your own using modeling software like Blender.

3. Scene Design: Create game scenes or levels where the simulation takes place. Design environments such as cities, forests, buildings, or any other relevant locations for a zombie apocalypse. Set up the terrain, structures, and other interactive elements.

4. Player and Survivor Controls: Implement player controls for movement, aiming, and interaction. Allow the player to control a survivor or multiple survivors in the game world. Implement mechanics for managing survivor attributes like health, stamina, hunger, and thirst.

5. Zombie AI: Implement artificial intelligence for the zombies. Define behaviors like wandering, chasing, attacking, and reacting to survivor actions. The AI should make the zombies appear intelligent and pose a threat to the survivors.

6. Survivor Network: Implement a data structure to represent the survivor network, such as a graph or a tree. The structure should maintain relationships between survivors, handle survivor interactions, and support dynamic updates as survivors join or leave groups.

7. Combat Mechanics: Implement combat mechanics for survivors to engage in battles with zombies. This may include shooting mechanics, melee combat, weapon management, and inventory systems. Consider factors like weapon damage, zombie strength, survivor skills, and resource management.

8. Resource Management: Implement systems for managing resources like food, water, ammunition, and medical supplies. Allow survivors to scavenge for resources, trade with other survivors, and use resources for survival and healing.

9. Events and Challenges: Create random events, missions, and challenges to add variety and unpredictability to the gameplay. These can include rescue missions, resource runs, defending bases, and encounters with other survivor groups or hostile humans.

10. Visual Effects and Audio: Enhance the immersion by adding visual effects, animations, sound effects, and background music. Implement lighting, particle effects, blood splatters, and other audio-visual elements to make the game more engaging and realistic.

11. Game Logic and Rules: Define the overall game logic and rules. Determine win and lose conditions, scoring mechanisms, and progression systems. Add features like day-night cycles, weather conditions, and environmental hazards to increase the challenge.

12. Playtesting and Iteration: Test the game extensively to identify and fix bugs, balance gameplay mechanics, and gather feedback. Iteratively improve the game based on playtesting results and user feedback.

Remember, building a realer zombie apocalypse simulation is a complex task that requires strong programming skills, game development knowledge, and possibly a team of developers, artists, and sound designers. This overview provides a starting point, but the actual implementation will require substantial effort and expertise in game development.
"""

"""
A binary tree or a similar data structure can be used in a zombie apocalypse simulation to represent various aspects of the scenario. Here are a few ways in which a binary tree can be utilized:

1. Survivor Network: Each node in the binary tree can represent a survivor in the simulation. The tree structure allows you to model relationships between survivors, such as family ties or group affiliations. Each node can store information about the survivor, such as their location, health status, available resources, and relationships with other survivors. The tree can be updated dynamically as survivors move, form alliances, or encounter each other.

2. Infection Spread: The binary tree can represent the spread of the zombie infection. Each node can represent an infected individual, and the child nodes can represent the people they have infected. This structure allows you to track the infection's progression, the spread rate, and the relationships between infected individuals. You can simulate the spread of the infection by traversing the tree and updating the infected nodes accordingly.

3. Resource Distribution: The binary tree can be used to model the distribution of resources, such as food, water, or weapons, among the survivors. Each node can represent a location or a group of survivors, and the child nodes can represent the allocation of resources within that location or group. You can simulate the movement of resources by redistributing them among the nodes during the simulation.

4. Decision Making: A binary tree can be employed to simulate decision-making processes for both survivors and zombies. Each node can represent a decision point, with the child nodes representing different choices or actions. For example, a survivor may face a decision to fight, hide, or scavenge for supplies. By traversing the tree based on the choices made, you can simulate the consequences of those decisions and the branching paths that result.

5. Event Sequencing: The binary tree can also be used to simulate the sequence of events during the zombie apocalypse. Each node can represent an event, such as a zombie attack, a rescue mission, or the discovery of a safe location. The child nodes can represent the subsequent events that occur as a result of the initial event. By traversing the tree, you can simulate the progression of events and their impact on the survivors and the overall scenario.

These are just a few examples of how a binary tree or a similar data structure can be utilized in a zombie apocalypse simulation. The specific implementation and use cases may vary depending on the requirements and goals of the simulation.
"""
