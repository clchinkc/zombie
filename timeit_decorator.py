from __future__ import annotations

import functools
import cProfile
import pstats
import timeit
import line_profiler
import memory_profiler

def performance_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Run the function using timeit.repeat and return the average execution time
        t = timeit.Timer(lambda: func(*args, **kwargs))
        res = t.repeat(repeat=10, number=100)
        time = min(res)
        print(f'Execution time of {func.__name__} is {time:.5f} seconds (minimum of 10 times with 100 runs each)')
        print()
        
        # Run the function using cProfile.runctx and return the profiling information
        profile_filename = func.__name__ + '.profile'
        cProfile.runctx('func(*args, **kwargs)', globals(), locals(), profile_filename)
        p = pstats.Stats(profile_filename)
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()

        # Run the function using line_profiler and return the line-by-line execution time
        lp = line_profiler.LineProfiler()
        lp.add_function(func)
        lp.enable_by_count()
        result = func(*args, **kwargs)
        lp.disable_by_count()
        lp.print_stats()

        # Run the function using memory_profiler and return the memory usage information
        memory_profiler.profile(func)(*args, **kwargs)

        return result
    return wrapper



class PerformanceDecorator:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def time_it(self, *args, **kwargs):
        # Run the function using timeit.repeat and return the average execution time
        t = timeit.Timer(lambda: self.func(*args, **kwargs))
        res = t.repeat(repeat=10, number=100)
        time = min(res)
        print(f'Execution time of {self.func.__name__} is {time:.5f} seconds (minimum of 10 times with 100 runs each)')
        print()

    def call_count(self, *args, **kwargs):
        # Run the function using cProfile.runctx and return the profiling information
        profile_filename = self.func.__name__ + '.profile'
        cProfile.runctx('self.func(*args, **kwargs)', globals(), locals(), profile_filename)
        p = pstats.Stats(profile_filename)
        p.strip_dirs()
        p.sort_stats('time')
        p.print_stats()

    def line_profile(self, *args, **kwargs):
        # Run the function using line_profiler and return the line-by-line execution time
        lp = line_profiler.LineProfiler()
        lp.add_function(self.func)
        lp.enable_by_count()
        result = self.func(*args, **kwargs)
        lp.disable_by_count()
        lp.print_stats()

    def memory_profile(self, *args, **kwargs):
        # Run the function using memory_profiler and return the memory usage information
        memory_profiler.profile(self.func)(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self.func(*args, **kwargs)



from collections import Counter

#@performance_decorator
def use_counter_to_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    for c in A2Z1000:
        d[c] += 1
    return d

#@performance_decorator
def counter_update_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    d.update(A2Z1000)
    return d

#@performance_decorator
def counter_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter(A2Z1000)
    return d

"""
# example 1
y = use_counter_to_count(1000)
print(y)
y = counter_update_count(1000)
print(y)
y = counter_count(1000)
print(y)
"""

"""
# example 2
decorator = PerformanceDecorator(use_counter_to_count)
decorator.time_it(1000)
decorator.call_count(1000)
decorator.line_profile(1000)
decorator.memory_profile(1000)

result = decorator.call(1000)
print(result)
"""

"""
https://myapollo.com.tw/zh-tw/cprofile-and-py-spy-tutorial/
https://myapollo.com.tw/zh-tw/fil-memory-usage-profiler/

https://myapollo.com.tw/zh-tw/python-concurrent-futures/
https://myapollo.com.tw/zh-tw/python-multiprocessing/
https://myapollo.com.tw/zh-tw/more-about-python-multiprocessing/
https://myapollo.com.tw/zh-tw/begin-to-asyncio/

"""
