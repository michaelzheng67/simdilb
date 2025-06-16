# Simdlib - SIMD-accelerated list operations in python

A case study into how NEON SIMD instrinsics on macos can accelerate list operations

## Why do we care?

Operations on lists of data can be found applicable in many computing applications, from finance
to LLMs to databases.

Moreover, Python is notorious for being slow due to many factors (interpreter, gc, abstraction
layers, etc) and being able to speed up python applications with minimal source changes
is a huge win.

## Why SIMD Intrinsics?

This library is for use by ARM architectures. Specifically, it utilizes the NEON SIMD extension
to provide it's capabilities. We choose to focus on SIMD intrinsics because:

- Reduces reliance on the compiler to produce performant code. Even compiler hints aren't
  guaranteed to produce the assembly we want

- Python C extension allows us to make lower-level optimizations to how our library works

Disadvantages of this approach include:

- Stuck to making platform-specific changes

- Error-prone / messier code

## How it works

Simdlib is written using the Python C Extension library. This allows us to cast python list objects
into their C counterparts. Each operation essentially follows the same process. We load in an
empty vector into a vector register. We then load 4 elements at a time into the vector register,
then use an explicit instruction to accumulate those values into a variable. Not every list is
going to have len(list) % 4 == 0, so with the remaining elements we accumulate them serially.

## Benchmark Results

Included in the the tests folder is benchmarking code. Running on a M3 Pro with Sequoia 15.5. Three
different methods were tested. We have the naive implementation, which is a serial for loop over
elements. We have our SIMD-accelerated implementation, which replaces the for loop with a library
call. Then, we also include a numpy example to show that in certain circumstances it can beat it
as well. We test out various accumulation operations on a nested list object.

I was able to get the following results (in secs) running 100 iterations each time:

```
sum op:
naive 0.6113339829444885
optimized 0.5228000903129577
numpy_func 0.8831670045852661

multiply op:
naive 0.6138886904716492
optimized 0.5130776786804199
numpy_func 0.8746641612052918

min op:
naive 0.6658641147613525
optimized 0.5254337954521179
numpy_func 0.894232234954834
209.55441308021545

max op:
naive 0.7053544044494628
optimized 0.5487733411788941
numpy_func 0.9244655680656433

any op:
naive 0.5905903220176697
optimized 0.5639568519592285
numpy_func 0.9229762864112854

all op:
naive 0.5866790390014649
optimized 0.5561843490600586
numpy_func 0.9211705374717712
```

In this specific case, we see that our optimized functions perform best. This is likely due to the
fact that versus the naive version, we utilize SIMD operations vs. performing each accumulate
step individually, and vs. our numpy implementation we can operate on native python lists vs
converting to np.arrays.

A "real-life" example is also available in tests/stock_example.py which demonstrates using simdlib
to calculate the sharpe ratio from a generated list of stock prices. The results are as follows:

naive 0.1298867130279541
optimized 0.10430599689483643
numpy 0.11439987182617188
