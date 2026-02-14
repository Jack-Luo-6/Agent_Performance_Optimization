import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    '''Setup - create test data targeting edge cases'''
    global df1, df2
    
    # Create DataFrames with conflicting indices and null values
    df1 = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['x', 'y', 'z']}, index=[0, 1, 2])
    df2 = pd.DataFrame({'A': [4, np.nan, 6], 'C': ['u', 'v', 'w']}, index=[2, 3, 4])

def workload():
    '''Workload - execute merge on DataFrames with edge cases'''
    global df1, df2
    
    result = pd.merge(df1, df2, on='A', how='outer')

# Benchmark (DO NOT MODIFY)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
