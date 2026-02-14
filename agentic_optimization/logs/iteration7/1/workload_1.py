import timeit
import statistics
# Add other required imports (MATCH the baseline imports!)
import pandas as pd
import numpy as np

def setup():
    '''Setup function - initialize test data here'''
    global test_data  # Declare all globals
    
    # Create test data
    # Make it HARDER than baseline if this is an adaptive iteration
    # Baseline targets: pandas DatetimeArray elementwise comparison (arr1 < arr2)
    # Strategy: 5x scale-up and diverse patterns (sequential, duplicates, alternating, random jitter, NaT edges)
    N = 50_000_000  # 5x the baseline size (10,000,000)

    # Base datetime index at 1-second frequency
    base = pd.date_range("2000-01-01", periods=N, freq="s")  # DatetimeIndex backbone
    arr1 = base._data  # DatetimeArray (matches baseline data structure)

    # Work with raw nanoseconds to build arr2 efficiently
    arr2_ns = arr1.asi8.copy()  # int64 nanoseconds
    one_sec = np.int64(1_000_000_000)
    iNaT = np.int64(np.iinfo(np.int64).min)
    s = N // 5

    # Insert NaT at both edges (0.5% on each side) to introduce edge-case comparisons
    edge = max(1, N // 200)
    arr2_ns[:edge] = iNaT
    arr2_ns[-edge:] = iNaT

    # Segment 1 (strictly greater): +1s
    arr2_ns[edge:edge + s] += one_sec

    # Segment 2 (equal): unchanged to force equality runs
    # arr2_ns[edge + s:edge + 2*s] remains the same

    # Segment 3 (strictly smaller): -1s
    arr2_ns[edge + 2*s:edge + 3*s] -= one_sec

    # Segment 4 (alternating): -2s for even indices, +3s for odd indices
    seg4_start = edge + 3*s
    seg4_end = edge + 4*s
    seg4 = slice(seg4_start, seg4_end)
    arr2_ns[seg4][::2] -= 2 * one_sec
    arr2_ns[seg4][1::2] += 3 * one_sec

    # Remaining region: sparse random jitter {-1,0,+1}s at 10% of positions
    rng = np.random.default_rng(2026)
    rem_start = edge + 4*s
    rem_end = N - edge
    if rem_end > rem_start:
        rem_len = rem_end - rem_start
        choose = max(1, rem_len // 10)
        idx = rng.choice(rem_len, size=choose, replace=False)
        jitter = rng.integers(-1, 2, size=choose, dtype=np.int64)  # -1,0,1
        arr2_ns[rem_start + idx] += jitter * one_sec

    # Convert back to DatetimeArray (same structure as baseline)
    arr2 = pd.to_datetime(arr2_ns, unit="ns")._data

    # Store both arrays for the workload
    test_data = (arr1, arr2)

def workload():
    '''Workload function - the actual work being tested'''
    global test_data  # Access globals
    
    # Call the target functions (SAME as baseline!)
    # Elementwise comparison on pandas DatetimeArray
    arr1, arr2 = test_data
    _ = arr1 < arr2

# Run benchmark (DO NOT MODIFY THIS SECTION)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
