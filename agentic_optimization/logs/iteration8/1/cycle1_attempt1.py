import timeit
import statistics
import pandas as pd
import numpy as np

def setup():
    '''Setup function - initialize test data here'''
    global test_data  # Declare all globals
    
    # Create test data
    # Make it HARDER than baseline if this is an adaptive iteration
    N = 50_000_000  # 5x baseline size

    # Construct int64 nanosecond arrays directly to minimize intermediate copies
    base_ns = pd.Timestamp("2000-01-01").value  # epoch ns for 2000-01-01
    step = np.int64(1_000_000_000)  # 1 second in ns
    v1 = base_ns + np.arange(N, dtype=np.int64) * step  # sequential seconds
    v2 = v1 + step  # offset by +1s (same semantics as baseline)

    # Inject duplicates with different periodic patterns for both arrays
    if N > 1:
        d1 = np.arange(1, N, 10, dtype=np.int64)
        v1[d1] = v1[d1 - 1]
        d2 = np.arange(2, N, 15, dtype=np.int64)
        v2[d2] = v2[d2 - 1]

    # Sprinkle NaT values at staggered positions (edge cases)
    NAT = np.iinfo(np.int64).min
    v1[::100] = NAT
    v2[50::100] = NAT

    # Create DatetimeArray objects (matching baseline)
    arr1 = pd.to_datetime(v1, unit="ns")._data  # DatetimeArray
    arr2 = pd.to_datetime(v2, unit="ns")._data  # DatetimeArray

    test_data = (arr1, arr2)  # Your data here

def workload():
    '''Workload function - the actual work being tested'''
    global test_data  # Access globals
    
    # Call the target functions (SAME as baseline!)
    arr1, arr2 = test_data
    arr1 < arr2

# Run benchmark (DO NOT MODIFY THIS SECTION)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
