
import math

from pyspark import SparkConf, SparkContext


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

if __name__ == "__main__":
    conf = SparkConf().setAppName("PrimeNumberFinder")
    sc = SparkContext(conf=conf)
    
    # Specify the range of numbers to search for primes
    lower_limit = 100000
    upper_limit = 200000
    
    # Create an RDD with the range of numbers
    numbers_rdd = sc.parallelize(range(lower_limit, upper_limit))
    
    # Filter the numbers RDD to keep only the prime numbers
    prime_numbers_rdd = numbers_rdd.filter(is_prime)
    
    # Collect the prime numbers into a list
    prime_numbers = prime_numbers_rdd.collect()
    
    # Print the prime numbers
    for prime in prime_numbers:
        print(prime)
    
    # Stop the Spark context
    sc.stop()

"""
Distributed systems are a fundamental part of modern computing, enabling the development of large-scale, reliable, and scalable applications. Here are some foundational concepts in distributed systems:

1. **Concurrency:** Concurrency refers to the ability of a distributed system to handle multiple tasks or processes simultaneously. It involves managing the execution of multiple independent units of computation, known as concurrent processes or threads, to maximize efficiency and throughput.

2. **Communication:** Communication is crucial in distributed systems, as components or nodes need to exchange information to coordinate their actions and achieve a common goal. Communication can happen through various mechanisms, such as message passing, remote procedure calls (RPC), snap-shot algorithms, or shared memory.

3. **Consistency:** Consistency deals with ensuring that all nodes in a distributed system agree on the same view of data, even when updates occur concurrently. Consistency models define the rules and guarantees regarding the order and visibility of read and write operations across different nodes.

4. **Fault tolerance:** Fault tolerance is the ability of a distributed system to continue functioning correctly in the presence of faults or failures. It involves mechanisms such as replication, redundancy, and error detection and recovery techniques to maintain system availability and data integrity.

5. **Scalability:** Scalability refers to the system's ability to handle increased workloads or growing numbers of users without significant performance degradation. Distributed systems employ various scalability techniques, including load balancing, partitioning, and distributed caching, to distribute the workload across multiple nodes efficiently.

6. **Distributed algorithms:** Distributed algorithms are algorithms designed to solve problems in distributed systems. They typically involve coordination, synchronization, and consensus protocols to achieve common objectives in a distributed and decentralized environment.

7. **Distributed file systems:** Distributed file systems provide a consistent and efficient mechanism for storing and accessing files across multiple nodes in a distributed system. They handle issues such as data distribution, replication, fault tolerance, and file consistency.

8. **Distributed transactions:** Distributed transactions involve coordinating multiple operations across multiple nodes in a distributed system, ensuring that they all succeed or fail together as a single atomic unit. Techniques like two-phase commit (2PC) or three-phase commit (3PC) protocols are used to achieve distributed transactional consistency.

9. **Eventual consistency:** Eventual consistency is a weaker consistency model that allows updates to propagate asynchronously in a distributed system. It guarantees that if no further updates occur, all replicas will eventually converge to the same state.

10. **Replication:** Replication involves creating and maintaining multiple copies of data across different nodes in a distributed system. It improves fault tolerance, data availability, and performance by allowing local access to data. Techniques like replication consistency protocols determine how updates are propagated and maintained across replicas.

These concepts provide a foundation for understanding the challenges and design principles behind distributed systems. However, it's important to note that distributed systems are a vast and evolving field, and there are many additional concepts, algorithms, and technologies that can be explored based on specific requirements and applications.
"""