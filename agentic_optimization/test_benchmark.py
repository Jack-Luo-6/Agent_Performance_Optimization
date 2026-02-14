import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    global df_large, df_varied
    df_large = pd.DataFrame(np.random.rand(100000, 10))
    df_varied = pd.DataFrame(np.random.randint(0, 100, size=(100000, 10)))

def workload():
    global df_large, df_varied
    result = pd.concat([df_large, df_varied])

# Benchmark
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}s")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}s")
