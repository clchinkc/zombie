
import asyncio
import random


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[None for _ in range(width)] for _ in range(height)]
        self.locks = [[asyncio.Lock() for _ in range(width)] for _ in range(height)]

class Entity:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
    
    async def move_to(self, new_x, new_y, grid, max_retries=3):
        # Check if the move is only one step in one of the specified directions
        if (new_x, new_y) not in [(self.x, self.y + 1), (self.x + 1, self.y), (self.x, self.y - 1), (self.x - 1, self.y)]:
            print(f"{self.name} tried to move in an invalid direction!")
            return
        
        for _ in range(max_retries):
            current_lock = grid.locks[self.y][self.x]
            target_lock = grid.locks[new_y][new_x]
            
            try:
                acquired = await asyncio.wait_for(target_lock.acquire(), timeout=0.2)
                if acquired:
                    if grid.cells[new_y][new_x] is None:
                        grid.cells[self.y][self.x] = None
                        await asyncio.sleep(0.1)
                        self.x, self.y = new_x, new_y
                        grid.cells[new_y][new_x] = self
                        print(f"{self.name} moved to ({new_x}, {new_y})")
                        target_lock.release()
                        return
                    target_lock.release()
            except asyncio.TimeoutError:
                pass
                
            await asyncio.sleep(0.1)  # Wait before retrying

        # If movement fails, move to a nearby location
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                await self.move_to(nx, ny, grid, max_retries=1)
                return

        print(f"{self.name} couldn't find a place to move!")

async def main():
    grid = Grid(5, 5)
    
    entity1 = Entity("Entity1", 1, 2)
    entity2 = Entity("Entity2", 3, 2)
    
    grid.cells[1][2] = entity1
    grid.cells[3][2] = entity2

    # Both entities try to move to (2, 2).
    await asyncio.gather(entity1.move_to(2, 2, grid), entity2.move_to(2, 2, grid))

asyncio.run(main())


"""
Additional Considerations:
You might want to introduce randomness in movement decisions to make the simulation more dynamic.
If there are many entities and frequent movements, you might encounter a situation known as "deadlock". It's a state where entities are waiting for each other to release locks indefinitely. Design your logic carefully to avoid such situations or consider adding timeouts to the locks.
Other synchronization primitives like asyncio.Semaphore or asyncio.Condition can be used depending on your specific requirements.
"""

"""
`asyncio`, `multiprocessing`, and `threading` are three different approaches in Python for managing concurrency and parallelism. Each approach has its own strengths and weaknesses, and the choice of which one to use depends on the specific use case and requirements of your application.

1. **Threading (`threading` module):**
Threading is a way to achieve concurrency in Python by running multiple threads within the same process. Threads share the same memory space, which can lead to potential issues like race conditions and deadlocks. Python's Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously in the same process, which can limit the parallelism achieved with threading. However, threading is useful for I/O-bound tasks where threads can be blocked waiting for I/O operations, allowing other threads to execute.

Pros:
- Good for I/O-bound tasks.
- Lightweight compared to multiprocessing.
- Threads can share data easily (but with care to avoid race conditions).

Cons:
- Limited parallelism due to the GIL.
- Not suitable for CPU-bound tasks that require true parallel execution.

2. **Multiprocessing (`multiprocessing` module):**
Multiprocessing is a way to achieve parallelism in Python by running multiple processes, each with its own memory space, interpreter, and GIL. This allows true parallel execution for CPU-bound tasks, as each process can utilize its own CPU core.

Pros:
- Suitable for CPU-bound tasks that require parallel execution.
- Each process has its own memory space, avoiding issues like the GIL.

Cons:
- Heavier memory consumption compared to threading.
- Inter-process communication (IPC) can be more complex than thread synchronization.

3. **Asyncio (`asyncio` module):**
Asyncio is a framework for writing asynchronous, non-blocking code using coroutines. It allows you to write single-threaded asynchronous code that can handle multiple tasks concurrently without the need for multiple threads or processes. Asyncio is well-suited for I/O-bound tasks where waiting for external resources (like network calls or file I/O) is the main bottleneck.

Pros:
- Efficient for I/O-bound tasks that require asynchronous behavior.
- Single-threaded model with cooperative multitasking.
- Can handle large numbers of concurrent connections with low overhead.

Cons:
- Not suitable for CPU-bound tasks that require parallel execution.
- Requires understanding of asynchronous programming concepts and coroutines.

In summary, choose the concurrency approach that best suits your specific use case:

- Use `threading` for I/O-bound tasks where concurrency can be achieved through waiting for I/O operations.
- Use `multiprocessing` for CPU-bound tasks that require true parallel execution across multiple CPU cores.
- Use `asyncio` for I/O-bound tasks where asynchronous programming can provide high concurrency without the overhead of creating multiple threads or processes.
"""

"""
Combining `threading`, `multiprocessing`, and `asyncio` in a zombie apocalypse simulation can help you create a realistic and efficient simulation. Here's how you could use each approach to model different aspects of the simulation:

1. **Threading (`threading` module):**
Use threading to manage concurrent I/O-bound tasks that involve interactions with the environment, such as:
- Simulating survivors' movements and actions.
- Interactions with non-zombie entities (e.g., finding resources, communicating with other survivors).
- Handling environmental changes (e.g., weather conditions, time progression).

Since these tasks involve waiting for I/O operations (like user input or external data), threading can help improve responsiveness and simulate multiple actions happening simultaneously.

2. **Multiprocessing (`multiprocessing` module):**
Leverage multiprocessing for CPU-bound tasks that can be parallelized, such as:
- Simulating the behavior of individual zombies and their interactions with the environment.
- Performing complex calculations related to pathfinding or zombie behavior.

By using multiprocessing, you can distribute the load across multiple CPU cores, effectively achieving true parallelism and speeding up the simulation for these computationally intensive tasks.

3. **Asyncio (`asyncio` module):**
Utilize asyncio to handle asynchronous interactions that involve waiting for I/O operations or events. This can include:
- Managing communication between survivor groups.
- Handling asynchronous events like survivors being attacked or finding shelter.

Asyncio's non-blocking nature allows you to efficiently simulate these interactions without creating a large number of threads or processes, resulting in better performance for I/O-bound tasks.

Overall, you can structure your simulation with a combination of these approaches:

- Use `threading` for tasks that involve I/O interactions with the environment and non-zombie entities.
- Use `multiprocessing` for CPU-bound tasks related to zombie behavior and complex calculations.
- Use `asyncio` for managing asynchronous events and interactions between survivor groups.

Keep in mind that combining these concurrency approaches requires careful design to avoid potential pitfalls like race conditions, deadlocks, or inefficient resource usage. Additionally, ensure that you're aware of the limitations and considerations of each approach to make the most appropriate choice for each aspect of your simulation.
"""
