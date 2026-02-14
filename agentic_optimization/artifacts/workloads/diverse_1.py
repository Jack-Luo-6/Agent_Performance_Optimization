import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    '''Setup - create test data targeting worst-case scenario'''
    global df_large, df_varied
    
    # Create large and varied DataFrame objects
    df_large = pd.DataFrame(np.random.rand(100000, 10))
    df_varied = pd.DataFrame(np.random.randint(0, 100, size=(100000, 10)))

def workload():
    '''Workload - execute concatenation of DataFrames'''
    global df_large, df_varied
    
    result = pd.concat([df_large, df_varied])

# Benchmark (DO NOT MODIFY)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
