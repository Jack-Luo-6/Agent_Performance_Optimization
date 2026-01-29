import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    global arr1, arr2
    
    N = 10_000_000
    
    base = pd.date_range("2000-01-01", periods=N, freq="s")
    arr1 = base._data  # DatetimeArray
    arr2 = pd.date_range("2000-01-01 00:00:01", periods=N, freq="s")._data

def workload():
    global arr1, arr2
    arr1 < arr2

runtimes = timeit.repeat(workload, number=10, repeat=25, setup=setup)
print("Mean:", statistics.mean(runtimes))
print("Std Dev:", statistics.stdev(runtimes))
