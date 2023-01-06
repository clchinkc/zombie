import functools
import time

class timeit:
  """decorator for benchmarking"""
  def __init__(self, fmt='completed {:s} in {:.5f} seconds'):
    # there is no need to make a class for a decorator if there are no parameters
    self.fmt = fmt
  
  def __call__(self, fn):
    # returns the decorator itself, which accepts a function and returns another function
    # wraps ensures that the name and docstring of 'fn' is preserved in 'wrapper'
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      # the wrapper passes all parameters to the function being decorated
      t1 = time.time()
      res = fn(*args, **kwargs)
      t2 = time.time()
      print(self.fmt.format(fn.__name__, t2-t1))
      return res
    return wrapper

from collections import Counter

@timeit()
def use_counter_to_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    for c in A2Z1000:
        d[c] += 1
    return d

@timeit()
def counter_count(times):
    A2Z1000 = ''.join(chr(i) for i in range(97, 123)) * times
    d = Counter()
    d.update(A2Z1000)
    return d


# example 1
y = use_counter_to_count(1000)
print(y)
y = counter_count(1000)
print(y)


# example 2
decorator = timeit(fmt='{:s} completed in {:.5f} seconds')

timed_use_counter_to_count = decorator(use_counter_to_count)
y = timed_use_counter_to_count(1000)
# print(y)

timed_counter_count = decorator(counter_count)
y = timed_counter_count(1000)
# print(y)


"""
https://myapollo.com.tw/zh-tw/cprofile-and-py-spy-tutorial/
https://myapollo.com.tw/zh-tw/fil-memory-usage-profiler/

https://myapollo.com.tw/zh-tw/python-concurrent-futures/
https://myapollo.com.tw/zh-tw/python-multiprocessing/
https://myapollo.com.tw/zh-tw/more-about-python-multiprocessing/
https://myapollo.com.tw/zh-tw/begin-to-asyncio/

"""
