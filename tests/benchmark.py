import time
import simdlib
import gc
import numpy as np
import math
import statistics
import functools
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--sum", help="Benchmark sum operations", action="store_true")
parser.add_argument("--mult", help="Benchmark multiply operations", action="store_true")
parser.add_argument("--min", help="Benchmark Min operations", action="store_true")
parser.add_argument("--max", help="Benchmark Max operations", action="store_true")
parser.add_argument("--any", help="Benchmark Any operations (OR on each element)", action="store_true")
parser.add_argument("--all", help="Benchmark All operations (AND on each element)", action="store_true")


args = parser.parse_args()



def timer(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if not hasattr(wrapper, "timings"):
      wrapper.timings = []
    start = time.time()
    func(*args, **kwargs)
    wrapper.timings.append(time.time() - start)

  return wrapper

N = 5_000

@timer
def naive():
  l = [[x for x in range(N)] for _ in range(N)]
  
  if args.sum:
    flatten_input = [x for ele in l for x in ele]
    sum(flatten_input)
  if args.mult:
    flatten_input = [x for ele in l for x in ele]
    math.prod(flatten_input)
  if args.min:
    flatten_input = [x for ele in l for x in ele]
    min(flatten_input)
  if args.max:
    flatten_input = [x for ele in l for x in ele]
    max(flatten_input)
  if args.any:
    flatten_input = [x for ele in l for x in ele]
    any(flatten_input)
  if args.all:
    flatten_input = [x for ele in l for x in ele]
    all(flatten_input)

@timer
def optimized():
  l = [[x for x in range(N)] for _ in range(N)]
  if args.sum:
    simdlib.sum_list(l)
  if args.mult:
    simdlib.multiply_list(l)
  if args.min:
    simdlib.min_list(l)
  if args.max:
    simdlib.max_list(l)
  if args.any:
    simdlib.any_list(l)
  if args.all:
    simdlib.all_list(l)

@timer
def numpy_func():
  input = np.array([[x for x in range(N)] for _ in range(N)])
  if args.sum:
    np.sum(input)
  if args.mult:
    np.prod(input)
  if args.min:
    np.min(input)
  if args.max:
    np.max(input)
  if args.any:
    np.any(input)
  if args.all:
    np.all(input)

def clear_cache():
	N = [x for x in range(100_000)]
	sum(N)
	gc.collect()

def benchmark_func(f, iterations):
  for i in range(iterations):
    clear_cache()
    f()
  print(f.__name__, statistics.mean(f.timings))

if __name__ == "__main__":
  iterations = 100

  with Pool(3) as p:
    p.starmap(benchmark_func, [(naive, iterations), (optimized, iterations), (numpy_func, iterations)])