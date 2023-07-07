import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def worker(num):
    """A simple CPU-bound worker function."""
    result = num * 2


def run_multiprocessing():
    """Run the worker function using multiprocessing."""
    pool = multiprocessing.Pool(processes=4)
    numbers = [1, 2, 3, 4, 5]
    start_time = time.time()
    pool.map(worker, numbers)
    end_time = time.time()
    pool.close()
    pool.join()
    return end_time - start_time


def run_threading():
    """Run the worker function using threading."""
    threads = []
    numbers = [1, 2, 3, 4, 5]
    start_time = time.time()
    for num in numbers:
        t = threading.Thread(target=worker, args=(num,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    return end_time - start_time


def run_concurrent_futures():
    """Run the worker function using concurrent.futures."""
    numbers = [1, 2, 3, 4, 5]
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
    # with ProcessPoolExecutor() as executor:
        executor.map(worker, numbers)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    multiprocessing_time = run_multiprocessing()
    threading_time = run_threading()
    concurrent_futures_time = run_concurrent_futures()

    print(f"Multiprocessing time: {multiprocessing_time} seconds")
    print(f"Threading time: {threading_time} seconds")
    print(f"Concurrent.futures time: {concurrent_futures_time} seconds")

"""
joblib vs dask vs mpi4py

When it comes to parallel programming in Python, the choice between multiprocessing, threading, and concurrent.futures depends on your specific requirements and the nature of your tasks. Here's a comparison of these approaches:

1. **Multiprocessing**:
   - Pros:
     - Allows you to take full advantage of multiple CPU cores for CPU-bound tasks by running separate processes.
     - Each process has its own memory space, which can enhance isolation and security.
     - Provides mechanisms like pipes, queues, and shared memory for inter-process communication.
   - Cons:
     - Creating and managing separate processes incurs some overhead.
     - Processes have higher memory overhead compared to threads.
     - Inter-process communication can be more complex than inter-thread communication.

2. **Threading**:
   - Pros:
     - Lightweight threads that are suitable for I/O-bound tasks or situations where you want to perform concurrent operations.
     - Threads share the same memory space, making data sharing easier.
     - Thread creation and management have lower overhead compared to processes.
   - Cons:
     - Due to the Global Interpreter Lock (GIL), threads in Python cannot fully utilize multiple CPU cores for CPU-bound tasks.
     - The GIL prevents native threads from executing Python bytecodes simultaneously, limiting the effectiveness of threading for CPU-bound tasks.
     - Thread synchronization and shared data access require careful consideration to avoid race conditions.

3. **concurrent.futures**:
   - Pros:
     - Provides a high-level interface for managing parallel tasks using both multiprocessing and threading.
     - Abstracts away the details of managing processes and threads, making it easier to write parallel code.
     - Allows you to switch between multiprocessing and threading with minimal code changes.
     - Provides features like futures, which represent the results of asynchronous computations.
   - Cons:
     - Similar to threading, the GIL limits the effectiveness of concurrent.futures for CPU-bound tasks.
     - You may still need to consider synchronization and shared data access when using multiple threads.

If you have CPU-bound tasks and want to fully utilize multiple CPU cores, multiprocessing is the most suitable option. However, it incurs some overhead due to process creation and communication mechanisms.

For I/O-bound tasks or situations where you need to perform concurrent operations, threading or concurrent.futures can be useful. concurrent.futures provides a higher-level interface and allows you to switch between multiprocessing and threading easily, depending on your requirements.

In general, if you have a choice between threading and multiprocessing, and your tasks are CPU-bound, multiprocessing is preferred. If your tasks are I/O-bound or require managing a large number of concurrent operations, threading or concurrent.futures can be a good fit.

Remember to consider factors like task characteristics, scalability, data sharing, synchronization, and the potential impact of the GIL when making a decision.
"""

"""
1. **Asynchronous**: Asynchronous programming refers to a programming paradigm where tasks are executed independently without waiting for the completion of previous tasks. It allows the program to perform other operations while waiting for certain tasks to complete. Asynchronous programming is often used to handle I/O-bound operations efficiently, such as network requests or file operations.

2. **Parallel**: Parallel programming involves the simultaneous execution of multiple tasks or operations. It aims to divide a large task into smaller sub-tasks that can be executed concurrently, typically on separate processors or CPU cores. Parallel programming enables faster execution by utilizing the available computational resources effectively. It is commonly used for CPU-bound tasks that can be divided into independent parts.

3. **Concurrent**: Concurrency refers to the ability of a program to execute multiple tasks or operations concurrently. It allows different tasks to make progress simultaneously, even if they are not executed in strict parallel. Concurrency is often achieved by interleaving tasks, switching between them quickly, and ensuring progress is made on multiple tasks. It is useful for managing multiple I/O-bound tasks, providing responsiveness, and improving overall system throughput.

4. **Multithreading**: Multithreading is a technique where multiple threads of execution are created within a single process. Each thread represents an independent sequence of instructions that can run concurrently with other threads. Multithreading allows different parts of a program to execute simultaneously, taking advantage of systems with multiple CPU cores. It is commonly used for concurrent and parallel programming, especially for applications with many I/O-bound tasks or where responsiveness is critical.
"""
