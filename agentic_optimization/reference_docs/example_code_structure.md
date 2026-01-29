# Example Code Structure Reference

## Main Functions

### process_data(items: list) -> list
Processes a list of items through transformation pipeline.
- Input: List of items (any type)
- Output: Processed list
- Performance: O(n) expected

### find_duplicates(items: list) -> list
Finds duplicate items in a list.
- Input: List of items
- Output: List of duplicates
- Performance: Currently O(n²), target O(n)

### compute_statistics(numbers: list) -> dict
Computes basic statistics on numbers.
- Input: List of numbers
- Output: Dict with mean, min, max, variance
- Performance: O(n) with multiple passes

## Expected Usage

```python
data = [1, 2, 3, 4, 5]
processed = process_data(data)
duplicates = find_duplicates(data)
stats = compute_statistics(data)
```

## Performance Targets

- Handle inputs up to 100,000 elements
- Sub-second processing for typical cases
- Memory efficient (< 100MB for large inputs)
