import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    '''Setup - create test data targeting memory pressure'''
    global large_array
    
    # Create a large tiled array to stress memory allocation
    small_array = np.random.rand(100, 100)
    large_array = np.tile(small_array, (1000, 1000))

def workload():
    '''Workload - execute operation on large tiled array'''
    global large_array
    
    # Mimic an operation that processes the large array
    result = np.sum(large_array)

# Benchmark (DO NOT MODIFY)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
