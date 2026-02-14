import timeit
import statistics
import pandas as pd
import numpy as np

def setup():
    '''Setup function - initialize test data here'''
    global test_data  # Declare all globals
    
    # Create test data
    # Make it HARDER than baseline if this is an adaptive iteration
    rng = np.random.default_rng(12345)

    # Large sequential pair (scaled up from 10M to 24M)
    N_seq = 24_000_000
    base_seq = pd.date_range("2000-01-01", periods=N_seq, freq="s")
    arr1_seq = base_seq._data  # DatetimeArray
    arr2_seq = pd.date_range("2000-01-01 00:00:01", periods=N_seq, freq="s")._data

    # Strided non-contiguous views (derived from sequential)
    arr1_stride = arr1_seq[1::2]
    arr2_stride = arr2_seq[1::2]

    # Pair with NaT values injected (pathological missing data)
    N_nat = 12_000_000
    base_nat1 = pd.date_range("2010-01-01", periods=N_nat, freq="s")._data.copy()
    base_nat2 = pd.date_range("2010-01-01 00:00:01", periods=N_nat, freq="s")._data.copy()
    # Inject ~0.5% NaT at different positions
    k_nat = max(1, N_nat // 200)
    idx1 = rng.choice(N_nat, size=k_nat, replace=False)
    idx2 = rng.choice(N_nat, size=k_nat, replace=False)
    base_nat1[idx1] = pd.NaT
    base_nat2[idx2] = pd.NaT

    # Randomly permuted order (non-monotonic, random patterns)
    N_rand = 16_000_000
    arr1_rand = pd.date_range("2020-01-01", periods=N_rand, freq="s")._data
    arr2_rand = pd.date_range("2020-01-01 00:00:01", periods=N_rand, freq="s")._data
    perm = rng.permutation(N_rand)
    arr1_rand = arr1_rand[perm]
    arr2_rand = arr2_rand[perm]

    # Duplicates-heavy pair (repeated values)
    N_dup_unique = 8_000_000
    dup_idx = pd.date_range("1990-01-01", periods=N_dup_unique, freq="s")
    dup_idx2 = dup_idx.append(dup_idx)  # duplicates by appending to itself -> length 16M
    arr1_dup = dup_idx2._data
    arr2_dup = pd.date_range("1990-01-01 00:00:01", periods=N_dup_unique * 2, freq="s")._data

    # Package all pairs as (name, arr1, arr2)
    test_data = [
        ("sequential_large", arr1_seq, arr2_seq),
        ("strided_views", arr1_stride, arr2_stride),
        ("with_nat", base_nat1, base_nat2),
        ("random_permuted", arr1_rand, arr2_rand),
        ("duplicates", arr1_dup, arr2_dup),
    ]

def workload():
    '''Workload function - the actual work being tested'''
    global test_data  # Access globals
    
    # Call the target functions (SAME as baseline!): vectorized DatetimeArray comparison (arr1 < arr2)
    for name, a, b in test_data:
        _ = a < b
        m = min(len(a), len(b))
        # Additional stress with varied slicing patterns but same operation
        _ = a[: m // 2] < b[: m // 2]
        _ = a[::-1] < b[::-1]
        if m > 2_000_000:
            _ = a[1_000_000 : m - 1_000_000] < b[1_000_000 : m - 1_000_000]

# Run benchmark (DO NOT MODIFY THIS SECTION)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
