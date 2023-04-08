from __future__ import annotations

import cProfile
import pstats
import timeit
from functools import update_wrapper, wraps

import line_profiler
import memory_profiler


def performance_decorator(mode='all'):
    def _performance_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if mode == 'time_it' or mode == 'all':
                # Run the function using timeit.repeat and return the average execution time
                t = timeit.Timer(lambda: func(*args, **kwargs))
                res = t.repeat(repeat=10, number=100)
                time = min(res) / 100
                try:
                    print(f'Execution time of method {func.__name__} from class {func.__qualname__.split(".")[0]} is {time:.5f} seconds')
                except AttributeError:
                    print(f'Execution time of function {func.__name__} is {time:.5f} seconds')
                print()
            
            if mode == 'call_count' or mode == 'all':
                # Create a profiler object
                profiler = cProfile.Profile()
                # Run the function and collect the profiler output
                profiler.enable()
                func(*args, **kwargs)
                profiler.disable()
                # Use pstats to process the profiler output
                p = pstats.Stats(profiler)
                p.strip_dirs()
                p.sort_stats('time')
                p.print_stats()
                """
                profile_filename = func.__name__ + '.profile'
                cProfile.runctx('func(*args, **kwargs)', globals(), locals(), profile_filename)
                p = pstats.Stats(profile_filename)
                p.strip_dirs()
                p.sort_stats('time')
                p.print_stats()
                """

            if mode == 'line_profile' or mode == 'all':
                # Run the function using line_profiler and return the line-by-line execution time
                lp = line_profiler.LineProfiler()
                lp.add_function(func)
                lp.enable_by_count()
                func(*args, **kwargs)
                lp.disable_by_count()
                lp.print_stats()
            
            if mode == 'memory_profile' or mode == 'all':
                # Run the function using memory_profiler and return the memory usage information
                memory_profiler.profile(func)(*args, **kwargs)
            
            return func(*args, **kwargs)
        return wrapper
    return _performance_decorator

"""
def time_count_decorator(init_func=None, *, time_unit='sec'):
    if init_func is None:
        return partial(time_count_decorator,time_unit=time_unit)

    @wraps(init_func)
    def time_count(*pos_args,**kw_args): 
        ''' The docstring of time_count '''  
        ts = time.time()
        return_value = init_func(*pos_args,**kw_args) 
        te = time.time()

        if time_unit == 'sec':
            time_used = te-ts

        elif time_unit == 'min':
            time_used = (te-ts)/60

        elif time_unit == 'hour':
            time_used = (te-ts)/60/60

        print ("{}'s time consume({}): {}".format(init_func.__name__,time_unit,time_used))
    
        return return_value  

    return time_count
    
@time_count_decorator #不想設置參數的時候
@time_count_decorator(time_unit='hour') #想設置參數的時候，必須用關鍵字參數設定
"""

class PerformanceDecorator:
    def __init__(self, func):
        self.func = func
        update_wrapper(self, func)

    def time_it(self, *args, **kwargs):
        # Run the function using timeit.repeat and return the average execution time
        t = timeit.Timer(lambda: self.func(*args, **kwargs))
        res = t.repeat(repeat=10, number=100)
        time = min(res) / 100
        print(f'Execution time of {self.func.__name__} is {time:.5f} seconds')
        print()

    def call_count(self, *args, **kwargs):
        # Create a profiler object
        profiler = cProfile.Profile()
        # Run the function and collect the profiler output
        profiler.enable()
        self.func(*args, **kwargs)
        profiler.disable()
        # Use pstats to process the profiler output
        p = pstats.Stats(profiler)
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

# performance decorator has 4 modes: time_it, call_count, line_profile, memory_profile

from collections import Counter


#@performance_decorator(mode='call_count')
def use_counter_to_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    for c in A2Z1000:
        d[c] += 1
    return d

#@performance_decorator(mode='all')
def counter_update_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    d.update(A2Z1000)
    return d

#@performance_decorator(mode='all')
def counter_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter(A2Z1000)
    return d

"""
# example 1
y = use_counter_to_count(1000)
print(y)
#y = counter_update_count(1000)
#print(y)
#y = counter_count(1000)
#print(y)
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

# If the decorated function is a method of a uneditable class
# and the decorator need to accept parameter, 
# can use dataclass for the class 
# and lock the decorator with the init variable using partial

"""
