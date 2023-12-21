import asyncio
import concurrent.futures
import random
import time


class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[None for _ in range(width)] for _ in range(height)]
        self.grid_lock = asyncio.Lock()

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    async def move_entity(self, entity, new_x, new_y):
        if not self.is_within_bounds(new_x, new_y) or not entity.is_valid_move(new_x, new_y):
            return False

        async with self.grid_lock:
            if self.cells[new_x][new_y] is None:
                self.cells[entity.x][entity.y] = None
                self.cells[new_x][new_y] = entity
                entity.x, entity.y = new_x, new_y
                return True
        return False

class Entity:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.override_decision = None

    def is_valid_move(self, new_x, new_y):
        return (new_x, new_y) in [(self.x, self.y), (self.x, self.y + 1), (self.x + 1, self.y), (self.x, self.y - 1), (self.x - 1, self.y)]

    async def act(self, grid, max_attempts=3):
        for _ in range(max_attempts):
            new_x, new_y = await self.make_decision()
            if await grid.move_entity(self, new_x, new_y):
                print(f"{self.name} moved to ({new_x}, {new_y})")
                return
            await asyncio.sleep(0.1)  # Brief pause before retrying
        print(f"{self.name} couldn't move after {max_attempts} attempts")

    async def make_decision(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            decision = await loop.run_in_executor(pool, self.complex_decision_logic)
            return decision

    def complex_decision_logic(self):
        time.sleep(random.uniform(0.3, 0.5))  # Simulate decision-making delay
        if self.override_decision:
            return self.override_decision
        move_x = random.choice([True, False])
        step = random.choice([-1, 1])
        new_x = self.x + step if move_x else self.x
        new_y = self.y + step if not move_x else self.y
        return new_x, new_y

async def main():
    grid = Grid(5, 5)
    entity1 = Entity("Entity1", 1, 2)
    entity2 = Entity("Entity2", 2, 3)

    # Place entities on the grid
    grid.cells[1][2] = entity1
    grid.cells[2][3] = entity2

    # Both entities try to move to (2, 2), but only entity1 succeeds.
    entity1.override_decision = (2, 2)
    entity2.override_decision = (2, 2)
    # Move both entities to (2, 2) in sequence to ensure entity1 moves to (2, 2) first
    await entity1.act(grid)
    await entity2.act(grid)

    # After entity1 moves to (2, 3), entity2 can move to (2, 2).
    entity1.override_decision = None
    entity2.override_decision = (2, 2)
    # Move both entities in parallel to test whether entity2 can move to the original position of entity1
    await asyncio.gather(entity1.act(grid), entity2.act(grid))
    
    entity1.override_decision = (2, 2)
    entity2.override_decision = None
    await asyncio.gather(entity1.act(grid), entity2.act(grid))

if __name__ == '__main__':
    asyncio.run(main())




"""
Additional Considerations:
Other synchronization primitives like asyncio.Condition can be used depending on your specific requirements.
"""

"""
import asyncio
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


# CPU-bound worker function
def cpu_worker(num):
    result = 1
    for _ in range(1000000):
        result += num * num
    return result

def write_to_file(filename, num):
    with open(filename, 'w') as f:
        for _ in range(10000):
            f.write(f"Some sample data for file {num}\n")

def read_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

# I/O-bound worker function
def io_worker(num):
    filename = f"tempfile_{num}.txt"
    # Writing to a file (simulates I/O operation)
    write_to_file(filename, num)

    # Reading from the same file (simulates I/O operation)
    data = read_from_file(filename)

    # Clean up by deleting the file after reading
    os.remove(filename)

    return len(data)

# Asynchronous CPU-bound worker function
async def async_cpu_worker(num):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, cpu_worker, num)

# Asynchronous I/O-bound worker function
async def async_io_worker(num):
    filename = f"tempfile_{num}.txt"

    loop = asyncio.get_running_loop()

    # Writing to a file
    await loop.run_in_executor(None, write_to_file, filename, num)

    # Reading from the same file
    data = await loop.run_in_executor(None, read_from_file, filename)

    # Clean up by deleting the file
    os.remove(filename)

    return len(data)


# Function to run tasks sequentially as a control
def run_sequentially(task, numbers):
    start_time = time.time()
    for num in numbers:
        task(num)
    end_time = time.time()
    return end_time - start_time

# Function to run tasks using multiprocessing
def run_with_multiprocessing(task, numbers):
    num_processes = multiprocessing.cpu_count()
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(task, numbers)
    end_time = time.time()
    return end_time - start_time

# Function to run tasks using ProcessPoolExecutor
def run_with_process_pool_executor(task, numbers):
    num_processes = multiprocessing.cpu_count()
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(task, numbers)
    end_time = time.time()
    return end_time - start_time

# Function to run tasks using threading
def run_with_threading(task, numbers):
    threads = []
    start_time = time.time()
    for num in numbers:
        t = threading.Thread(target=task, args=(num,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    return end_time - start_time

# Function to run tasks using ThreadPoolExecutor
def run_with_thread_pool_executor(task, numbers):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=len(numbers)) as executor:
        executor.map(task, numbers)
    end_time = time.time()
    return end_time - start_time

# Function to run asynchronous tasks
async def run_asyncio(tasks):
    start_time = time.time()
    await asyncio.gather(*tasks)
    end_time = time.time()
    return end_time - start_time

# Main execution
if __name__ == "__main__":
    numbers = list(range(1, 100))

    # Testing CPU-bound tasks
    print("CPU-bound tasks:")
    print(f"Sequential time: {run_sequentially(cpu_worker, numbers)} seconds")
    print(f"Multiprocessing time: {run_with_multiprocessing(cpu_worker, numbers)} seconds")
    print(f"ProcessPoolExecutor time: {run_with_process_pool_executor(cpu_worker, numbers)} seconds")
    print(f"Threading time: {run_with_threading(cpu_worker, numbers)} seconds")
    print(f"ThreadPoolExecutor time: {run_with_thread_pool_executor(cpu_worker, numbers)} seconds")
    print(f"Asyncio time: {asyncio.run(run_asyncio([async_cpu_worker(num) for num in numbers]))} seconds")

    # Testing I/O-bound tasks
    print("\nI/O-bound tasks:")
    print(f"Sequential time: {run_sequentially(io_worker, numbers)} seconds")
    print(f"Multiprocessing time: {run_with_multiprocessing(io_worker, numbers)} seconds")
    print(f"ProcessPoolExecutor time: {run_with_process_pool_executor(io_worker, numbers)} seconds")
    print(f"Threading time: {run_with_threading(io_worker, numbers)} seconds")
    print(f"ThreadPoolExecutor time: {run_with_thread_pool_executor(io_worker, numbers)} seconds")
    print(f"Asyncio time: {asyncio.run(run_asyncio([async_io_worker(num) for num in numbers]))} seconds")

# Multiprocessing and ProcessPoolExecutor are faster for CPU-bound tasks
# Threading and ThreadPoolExecutor are faster for I/O-bound tasks

# CPU-bound tasks:
# Sequential time: 5.640041351318359 seconds
# Multiprocessing time: 2.280035972595215 seconds
# ProcessPoolExecutor time: 2.140997886657715 seconds
# Threading time: 5.285039901733398 seconds
# ThreadPoolExecutor time: 5.435959815979004 seconds
# Asyncio time: 5.488828659057617 seconds

# I/O-bound tasks:
# Sequential time: 1.822998285293579 seconds
# Multiprocessing time: 1.2620019912719727 seconds
# ProcessPoolExecutor time: 1.740999698638916 seconds
# Threading time: 1.291999340057373 seconds
# ThreadPoolExecutor time: 1.2890028953552246 seconds
# Asyncio time: 1.3189995288848877 seconds
"""

"""
joblib vs dask vs mpi4py
"""

"""
1. **Asynchronous**: Asynchronous programming refers to a programming paradigm where tasks are executed independently without waiting for the completion of previous tasks. It allows the program to perform other operations while waiting for certain tasks to complete. Asynchronous programming is often used to handle I/O-bound operations efficiently, such as network requests or file operations.

2. **Parallel**: Parallel programming involves the simultaneous execution of multiple tasks or operations. It aims to divide a large task into smaller sub-tasks that can be executed concurrently, typically on separate processors or CPU cores. Parallel programming enables faster execution by utilizing the available computational resources effectively. It is commonly used for CPU-bound tasks that can be divided into independent parts.

3. **Concurrent**: Concurrency refers to the ability of a program to execute multiple tasks or operations concurrently. It allows different tasks to make progress simultaneously, even if they are not executed in strict parallel. Concurrency is often achieved by interleaving tasks, switching between them quickly, and ensuring progress is made on multiple tasks. It is useful for managing multiple I/O-bound tasks, providing responsiveness, and improving overall system throughput.

4. **Multithreading**: Multithreading is a technique where multiple threads of execution are created within a single process. Each thread represents an independent sequence of instructions that can run concurrently with other threads. Multithreading allows different parts of a program to execute simultaneously, taking advantage of systems with multiple CPU cores. It is commonly used for concurrent and parallel programming, especially for applications with many I/O-bound tasks or where responsiveness is critical.
"""



"""
For your server-client zombie apocalypse simulation, using a combination of `asyncio` for handling client connections and `multiprocessing` for executing CPU-bound tasks offers a robust and efficient solution:

1. **Client Connections with Asyncio:**
   - **Asyncio** is ideal for managing client connections due to its asynchronous, non-blocking nature. It allows the server to handle multiple client requests concurrently without the overhead of threading.
   - This approach is particularly efficient for I/O-bound operations, such as receiving updates from clients, sending responses back, and managing network communication.
   - Using asyncio, you can design a server that's capable of maintaining high responsiveness and scalability, even with a large number of simultaneous client connections.

2. **CPU-Bound Tasks with Multiprocessing:**
   - **Multiprocessing** provides true parallel execution by running separate processes, bypassing the limitations of Python's Global Interpreter Lock (GIL). 
   - It's well-suited for CPU-intensive tasks that your simulation might demand, like complex AI computations for zombies, physics simulations, or any other heavy computational logic.
   - Each process can run on a separate CPU core, significantly improving performance for these demanding tasks.
   - Although multiprocessing introduces a higher memory footprint and requires careful handling of inter-process communication, it ensures that your CPU-bound tasks are executed efficiently.

**Combined Workflow:**
- Your server uses asyncio to efficiently handle and respond to client requests in a non-blocking manner. This ensures that the server remains responsive to user actions and network events.
- When the server encounters a CPU-intensive task (like AI logic or simulation calculations), it offloads this task to a separate process managed by the multiprocessing module.
- The multiprocessing approach ensures that these tasks are executed in parallel, making full use of the server's multi-core CPU architecture, thus improving overall performance and responsiveness.
- Communication between the asyncio event loop (handling client connections) and the multiprocessing tasks (handling CPU-bound logic) should be well-structured, possibly using queues or other IPC mechanisms to coordinate and exchange data.

This combined approach leverages the strengths of both asyncio and multiprocessing, leading to a server that can efficiently handle a large number of client interactions while simultaneously performing complex computational tasks.
"""
