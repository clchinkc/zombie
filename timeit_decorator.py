import functools
import time

class timeit:
  """decorator for benchmarking"""
  def __init__(self, fmt='completed {:s} in {:.3f} seconds'):
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

# example 1: 
@timeit()
def test(msg='hello world', n=1):
  for _ in range(n):
    print(msg)

test(n=5)

# example 2
def sumupto(n: int):
  return sum(range(n))

decorator = timeit(fmt='{:s} completed in {:.3f} seconds')
timed_sumupo = decorator(sumupto)
y = timed_sumupo(10**7)
print(y)
