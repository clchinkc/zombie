
from abc import ABC, abstractmethod


class InputComponent(ABC):
    @abstractmethod
    def update(self, obj):
        pass

class PlayerInputComponent(InputComponent):

    def __init__(self) -> None:
        self.WALK_ACCELERATION = 1

    def update(self, obj):
        if Controller.getJoystickDirection() == DIR_LEFT:
            obj.velocity -= self.WALK_ACCELERATION
        elif Controller.getJoystickDirection() == DIR_RIGHT:
            obj.velocity += self.WALK_ACCELERATION


class DemoInputComponent(InputComponent):
    # AI to automatically control Bjorn
    def update(self, obj):
        pass

class PhysicsComponent(ABC):
    @abstractmethod
    def update(self, obj, world):
        pass

class BjornPhysicsComponent(PhysicsComponent):
    
    def __init__(self, volume) -> None:
        self.volume = volume
    
    def update(self, obj, world):
        obj.x += obj.velocity
        world.resolveCollision(self.volume, obj.x, obj.y, obj.velocity)

class GraphicsComponent:
    def update(self, obj, graphics):
        pass

class BjornGraphicsComponent(GraphicsComponent):
    
    def __init__(self, spriteStand, spriteWalkLeft, spriteWalkRight) -> None:
        self.spriteStand = spriteStand
        self.spriteWalkLeft = spriteWalkLeft
        self.spriteWalkRight = spriteWalkRight
    
    def update(self, obj, graphics):
        if obj.velocity < 0:
            sprite = self.spriteWalkLeft
        elif obj.velocity > 0:
            sprite = self.spriteWalkRight
        else:
            sprite = self.spriteStand
        graphics.draw(sprite, obj.x, obj.y)

class GameObject:
    def __init__(self, input, physics, graphics):
        self.velocity = 0
        self.x = 0
        self.y = 0
        self.input = input
        self.physics = physics
        self.graphics = graphics

    def update(self, world, graphics):
        self.input.update(self)
        self.physics.update(self, world)
        self.graphics.update(self, graphics)

def createBjorn():
    return GameObject(PlayerInputComponent(), 
                        BjornPhysicsComponent(), 
                        BjornGraphicsComponent())

"""
User inputcomponent that has method using Character as input to update the Character
Update the inputcomponent inside the Character update method
The Character own the inputcomponent that control the Character velocity by a specific acceleration stored inside inputcomponent
The character also own a physicscomponent that change the Character position according to its velocity and world.resolve collision function with the character and world as input
The character also own a graphicscomponent that points to left if velocity lower than 0 else right and graphic.draw function with character and graphics as input
So the character has position, velocity, inputcomponent, physicscomponent, graphicscomponent and update function that call input.update, physics.update, graphics.update
It holds components and state shared among components
"""

"""
How do components communicate with each other?

By modifying the container object's state: shared state in parent object

By referring directly to each other: but coupled

By sending messages: fire-and-forget messaging component in each component
"""
"""
Unsurprisingly, there's no one best answer here. What you'll likely end up doing is using a bit of all of them. Shared state is useful for the really basic stuff that you can take for granted that every object has — things like position and size.

Some domains are distinct but still closely related. Think animation and rendering, user input and AI, or physics and collision. If you have separate components for each half of those pairs, you may find it easiest to just let them know directly about their other half.

Messaging is useful for “less important” communication. Its fire-and-forget nature is a good fit for things like having an audio component play a sound when a physics component sends a message that the object has collided with something.

As always, I recommend you start simple and then add in additional communication paths if you need them.
"""
"""
https://gameprogrammingpatterns.com/component.html
# https://docs.gamecreator.io/gamecreator/characters/component/
"""


"""
Designing an MMORPG game where all agents are controlled by simple logic requires a combination of clean, modular, and optimized functional programming and object-oriented programming.


Functional programming emphasizes the use of functions and immutable data structures to build programs that are easy to reason about, while object-oriented programming emphasizes the use of objects and their interactions to build programs that are easy to extend and maintain.

To combine these two programming paradigms, we can use functional programming to define the core logic of the game, and use object-oriented programming to implement the agents that interact with that logic.

Here is a high-level overview of how this could be accomplished:

Define the game world as a data structure that consists of tiles and objects that can be interacted with.

Define the game logic as a set of pure functions that take the game world as input and return a new game world as output.

Define the agents as objects that have a set of behaviors and a reference to the current game world.

Each agent is controlled by a simple decision-making algorithm that determines which behavior to use based on the current game state and the agent's objectives.

The agent's behaviors are implemented as methods that modify the game world using the game logic functions.

Use object-oriented programming to create a hierarchy of agent classes that inherit from a base agent class. This allows for easy customization and extension of agent behavior.

Use functional programming techniques to ensure that the game logic is modular, testable, and efficient.

Use object-oriented programming techniques to ensure that the agent objects are easy to extend, maintain, and integrate with the game logic.

By combining functional programming and object-oriented programming, we can create an MMORPG game where all agents are controlled by simple logic that is easy to reason about and extend. This approach also allows for efficient use of resources and scalability.
"""

"""
Design the component of the agent class and explain how each components interact with each other or extra libraries.


An agent is an intelligent software system that can perceive its environment, reason about it, and take actions to achieve specific goals. In order to create an agent class in software development, there are several components that need to be designed and implemented. These components are:

Environment: The environment component defines the external world in which the agent operates. It provides the agent with inputs (i.e., sensory information) and receives outputs (i.e., actions) from the agent. The environment can be implemented as a separate class or as a module within the agent class.

Perception: The perception component takes sensory inputs from the environment and converts them into a form that the agent can understand. For example, if the agent is designed to play a game, the perception component may take in visual information from the game screen and convert it into a format that the agent can process.

Reasoning: The reasoning component uses the information provided by the perception component to make decisions and plan actions. It may use a variety of reasoning techniques such as logic, probabilistic reasoning, or machine learning algorithms.

Action: The action component takes the decisions made by the reasoning component and executes them by sending commands to the environment. For example, if the agent is designed to play a game, the action component may send commands to the game engine to move the player character.

Learning: The learning component allows the agent to improve its performance over time by adapting to changing environments or by learning from past experiences. This can be done using a variety of learning algorithms such as reinforcement learning, supervised learning, or unsupervised learning.

These components interact with each other in various ways. For example, the perception component provides input to the reasoning component, which in turn provides output to the action component. The action component then sends commands to the environment, which provides new sensory input to the perception component, and the cycle continues. The learning component can also interact with the other components by modifying their behavior based on past experiences.

In addition to these components, an agent class may also interact with extra libraries or tools such as databases, APIs, or external hardware devices. For example, an agent designed to control a robot may need to interact with sensors and actuators connected to the robot hardware. This interaction can be achieved using libraries or APIs that provide a standardized interface for the agent to communicate with the hardware. Similarly, an agent that needs to store and retrieve large amounts of data may need to interact with a database system to efficiently manage the data.
"""
