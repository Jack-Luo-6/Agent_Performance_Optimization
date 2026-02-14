import timeit
import statistics
# Add other required imports (MATCH the baseline imports!)
import pandas as pd
import numpy as np

def setup():
    """Setup function - initialize test data here"""
    global test_data  # Declare all globals

    # Create test data
    # Make it HARDER than baseline if this is an adaptive iteration
    # Baseline targets: pandas DatetimeArray elementwise comparison (arr1 < arr2)
    # Strategy: 5x scale-up and diverse patterns (sequential, duplicates, alternating, random jitter)
    N = 50_000_000  # 5x the baseline size (10,000,000)

    # Base datetime index
    base = pd.date_range("2000-01-01", periods=N, freq="s")  # DatetimeIndex backbone

    # Prepare arr2 by modifying a copy of base values in-place to avoid multiple huge temporaries
    arr2_vals = base.values.copy()  # dtype=datetime64[ns]

    s = N // 5

    # Segment 1: strictly greater (shift by +1s)
    arr2_vals[:s] += np.timedelta64(1, "s")

    # Segment 2: equal values (duplicates) -> unchanged
    # arr2_vals[s:2*s] remains the same

    # Segment 3: strictly smaller (shift by -1s)
    arr2_vals[2*s:3*s] += np.timedelta64(-1, "s")

    # Segment 4: alternating -2s, +3s
    seg4_len = (4*s) - (3*s)
    alt = np.empty(seg4_len, dtype="timedelta64[s]")
    alt[::2] = np.timedelta64(-2, "s")
    alt[1::2] = np.timedelta64(3, "s")
    arr2_vals[3*s:4*s] += alt

    # Segment 5: random small jitter in {-1, 0, +1} seconds
    rng = np.random.default_rng(2024)
    len5 = N - 4*s
    jitter = rng.integers(-1, 2, size=len5, dtype=np.int64)  # -1,0,1
    arr2_vals[4*s:] += jitter * np.timedelta64(1, "s")

    # Prepare DatetimeArray objects (same as baseline access pattern)
    arr1 = base._data  # DatetimeArray
    arr2 = pd.DatetimeIndex(arr2_vals)._data  # DatetimeArray

    # Store both arrays for the workload
    test_data = (arr1, arr2)

def workload():
    """Workload function - the actual work being tested"""
    global test_data  # Access globals

    # Call the target functions (SAME as baseline!)
    # Elementwise comparison on pandas DatetimeArray
    arr1, arr2 = test_data
    _ = arr1 < arr2

# Run benchmark (DO NOT MODIFY THIS SECTION)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
