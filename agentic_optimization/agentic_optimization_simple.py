import numpy as np
import timeit
import statistics

def maybe_sort(data, cache=None):
    if cache is not None and np.array_equal(data, cache['data']):
        return cache['sorted_data']
    sorted_data = np.sort(data)
    if cache is not None:
        cache['data'] = np.copy(data)
        cache['sorted_data'] = np.copy(sorted_data)
    return sorted_data

def workload_with_optimization():
    global large_data, edge_cases
    cache = {'data': None, 'sorted_data': None}
    for data in [large_data] + edge_cases:
        _ = maybe_sort(data, cache)

def test_workload_with_optimizations():
    global large_data, complex_data, edge_cases
    large_data = np.random.randint(0, 1000, 10000).tolist()  # Reduced data size
    complex_data = [{'key': np.random.randint(0, 100, 100).tolist()} for _ in range(1000)]  # Reduced complexity size
    edge_cases = [
        [0] * 10000,  # Reduced size for edge cases
        list(range(10000)),
        list(range(10000, 0, -1)),
        []
    ]

    runtimes = timeit.repeat(workload_with_optimization, repeat=2, number=1)  # Reduced repetitions
    print(f"Optimized Mean: {statistics.mean(runtimes):.6f}")
    print(f"Optimized Std Dev: {statistics.stdev(runtimes):.6f}")

# Trigger tests
test_workload_with_optimizations()
