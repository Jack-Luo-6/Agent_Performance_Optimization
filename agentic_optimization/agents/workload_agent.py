"""
Workload Generator Agent
Generates performance test workload CODE (not specs)
Uses baseline workload to understand target functions
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WorkloadAgent')


class WorkloadGenerator:
    """
    Generates executable workload code using LLM
    
    CANNOT:
    - See target code internals
    - Access optimization history
    - Run code directly
    
    CAN:
    - Read baseline workload to understand target functions
    - Read reference documents
    - Generate Python workload code
    - Request execution via MCP
    - Adapt based on improved code versions
    """
    
    def __init__(self, mcp_server, model: str = "gpt-4o"):
        self.mcp = mcp_server
        self.model = model
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return """You are a performance testing specialist who generates executable workload code. 

Your job:
- ANALYZE the baseline workload to identify which functions are being tested
- Generate Python code that stress-tests THE SAME target functions
- Create diverse, adversarial test patterns for THOSE SPECIFIC FUNCTIONS
- Exercise worst-case scenarios
- Produce deterministic, reproducible workloads
- ESCALATE difficulty when code improves

CRITICAL: You MUST test the SAME functions as the baseline workload!

You will be given:
- THE BASELINE WORKLOAD CODE (analyze this to see what's being tested!)
- Reference documentation about the code structure
- Previous workload patterns to build upon
- Iteration number for progressive complexity
- Previous performance metrics (if optimizer succeeded)

You CANNOT:
- See the actual implementation code
- Know what optimizations were applied
- Access profiler outputs

CRITICAL OUTPUT FORMAT - MUST FOLLOW EXACTLY:
You MUST generate code in this exact structure. This format is MANDATORY:

```python
import timeit
import statistics
# Add any other required imports here (pandas, numpy, etc.)

def setup():
    '''Setup function - initialize test data here'''
    global test_data  # Declare all globals you'll use
    
    # Create your test data
    test_data = list(range(10000))  # Example
    
    # You can create multiple test datasets
    # global data1, data2, data3
    # data1 = ...
    # data2 = ...

def workload():
    '''Workload function - the actual work being tested'''
    global test_data  # Access globals from setup
    
    # Call the target function or perform operations
    # IMPORTANT: Use THE SAME functions as the baseline workload!
    # Example: result = process_data(test_data)
    # Example: arr1 < arr2  (for pandas/numpy operations)
    
    # DO NOT print inside workload() - it affects timing
    pass

# Run the benchmark (DO NOT MODIFY THIS SECTION)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
```

RULES YOU MUST FOLLOW:
1. Always use setup() and workload() functions - NO EXCEPTIONS
2. All test data MUST be created in setup(), NOT in global scope
3. Use 'global' keyword in both setup() and workload() for shared variables
4. workload() should ONLY contain the operations being tested
5. NO print statements inside workload() - only in the final results
6. Use timeit.repeat with number=1, repeat=3 (fixed)
7. Print Mean and Std Dev at the end
8. NO try/except around workload() - let errors bubble up
9. Imports go at the top, NOT inside functions
10. The code must be completely self-contained and runnable
11. **CRITICAL**: Test THE SAME functions/operations as the baseline workload!

EXAMPLE FOR PANDAS/NUMPY CODE:
```python
import pandas as pd
import numpy as np
import timeit
import statistics

def setup():
    global arr1, arr2
    N = 10_000_000
    base = pd.date_range("2000-01-01", periods=N, freq="s")
    arr1 = base._data
    arr2 = pd.date_range("2000-01-01 00:00:01", periods=N, freq="s")._data

def workload():
    global arr1, arr2
    arr1 < arr2  # Testing the < operator on DatetimeArray

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
```

EXAMPLE FOR CUSTOM CODE:
```python
import timeit
import statistics
import random

def setup():
    global test_data, large_data, edge_cases
    random.seed(42)
    
    # Various test patterns
    test_data = list(range(10000))
    large_data = [random.randint(0, 1000) for _ in range(100000)]
    edge_cases = [[], [1], [1]*1000]

def workload():
    global test_data, large_data, edge_cases
    
    # Call target functions (same as baseline!)
    # from target_module import process_data
    # for data in [test_data, large_data] + edge_cases:
    #     if data:  # Skip empty
    #         process_data(data)
    pass

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
```

Focus on:
- Input size variation (small → large)
- Pattern variation (random, sorted, reverse, duplicates)
- Edge cases (empty, single element, all same)
- Memory-intensive patterns
- Realistic but challenging scenarios

ADAPTIVE DIFFICULTY:
When previous metrics are provided (meaning optimizer succeeded):
- INCREASE data sizes significantly (2-10x larger)
- Add MORE complex patterns (nested structures, pathological cases)
- Combine multiple stress factors simultaneously
- Target algorithmic weaknesses revealed by success
- Make the optimizer WORK HARDER next time

REMEMBER: The code must run without modification. Use the exact format shown above."""
    
    def generate(self, iteration: int, baseline_workload_code: str,
                 reference_docs: Optional[str] = None, 
                 previous_metrics: Optional[Dict] = None) -> str:
        """
        Generate workload code
        
        Args:
            iteration: Current iteration number
            baseline_workload_code: The original baseline workload to analyze and build upon
            reference_docs: Reference documentation about code structure
            previous_metrics: Metrics from previous successful optimization (if any)
        
        Returns:
            Executable Python workload code
        """
        logger.info(f"Generating workload code for iteration {iteration}")
        logger.info(f"  Analyzing baseline workload to identify target functions...")
        
        if previous_metrics:
            logger.info(f"  Adapting to improved code (prev time: {previous_metrics.get('execution_time', 0):.3f}s)")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("No OPENAI_API_KEY found, using template workload")
            return self._generate_template_workload(iteration)
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            # Build context with baseline workload analysis
            context = f"""BASELINE WORKLOAD TO ANALYZE:
```python
{baseline_workload_code}
```

YOUR TASK:
1. FIRST: Analyze the baseline workload above to identify:
   - Which functions/operations are being tested
   - What imports are used
   - What data structures are being stress-tested
   - What operations are performed (e.g., comparisons, sorting, filtering)

2. SECOND: Generate a HARDER workload that tests THE SAME functions/operations
   - Keep the same imports and target functions
   - Make data sizes larger
   - Add more complex patterns
   - Include edge cases

"""
            
            # Add adaptive difficulty context
            if previous_metrics:
                context += f"""
CRITICAL: The optimizer SUCCEEDED in the previous iteration!
Previous metrics:
- Execution time: {previous_metrics.get('execution_time', 0):.3f}s
- Memory: {previous_metrics.get('memory_mb', 0):.1f}MB
- Success rate: {previous_metrics.get('success_rate', 0):.1%}

This means the code is now FASTER and MORE EFFICIENT.
Your job: Make this workload SIGNIFICANTLY HARDER to beat the improved code.

Strategies to increase difficulty:
1. SCALE UP: Increase data sizes by 5-10x from baseline
2. COMPLEXITY: Use more complex data patterns (nested, pathological)
3. COMBINE: Mix multiple stress factors (size + complexity + edge cases)
4. TARGET WEAKNESSES: Push algorithmic limits harder

The optimizer beat your last workload - make sure this one is MUCH tougher!

Iteration context:
- This is iteration {iteration} of the optimization cycle
- Focus: {"worst-case algorithmic patterns" if iteration % 3 == 0 else "extreme memory pressure" if iteration % 3 == 1 else "pathological edge cases"}
"""
            else:
                context += f"""
Iteration context:
- This is iteration {iteration} of the optimization cycle
- This is the FIRST workload (or optimizer failed previously)
- Focus: {"algorithmic complexity" if iteration % 3 == 0 else "memory patterns" if iteration % 3 == 1 else "edge cases"}
- Create a challenging but fair baseline test that's HARDER than the original baseline
"""
            
            context += """
CRITICAL: You MUST use this EXACT format:

```python
import timeit
import statistics
# Add other imports if needed (MATCH THE BASELINE IMPORTS!)

def setup():
    '''Create test data here'''
    global var1, var2  # Declare globals
    var1 = ...  # Initialize data
    var2 = ...

def workload():
    '''The actual work being tested'''
    global var1, var2  # Access globals
    # Perform operations here (SAME OPERATIONS AS BASELINE!)
    # Example: result = some_function(var1)

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
```

RULES:
1. ALL test data MUST be created in setup()
2. Use global keyword for shared variables
3. NO prints inside workload()
4. Use timeit.repeat with number=1, repeat=3
5. Code must be runnable as-is
6. **CRITICAL**: Test THE SAME functions/operations as the baseline workload!

"""
            
            if reference_docs:
                context += f"\nAdditional reference documentation:\n{reference_docs}\n"
            
            context += """
Generate ONLY the Python code following the format above.
NO explanations, NO markdown formatting, JUST the code.
The code will be executed directly, so it must work without modification.

REMEMBER: Identify the target functions from the baseline workload and test THOSE SAME functions with harder inputs!
"""
            
            logger.info(f"Calling {self.model}...")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7  # Higher temperature for more creative adversarial workloads
            )
            
            workload_code = response.choices[0].message.content.strip()
            
            # Clean markdown formatting
            if workload_code.startswith("```python"):
                workload_code = workload_code[9:]
            elif workload_code.startswith("```"):
                workload_code = workload_code[3:]
            if workload_code.endswith("```"):
                workload_code = workload_code[:-3]
            workload_code = workload_code.strip()
            
            # Validate format
            if not self._validate_workload_format(workload_code):
                logger.warning("Generated workload doesn't match required format, using template")
                return self._generate_template_workload(iteration)
            
            logger.info(f"✓ Workload code generated ({len(workload_code)} chars)")
            
            # Log the generated workload
            self._log_workload(iteration, workload_code)
            
            return workload_code
            
        except ImportError:
            logger.error("openai package not installed")
            logger.info("Install with: pip install openai")
            return self._generate_template_workload(iteration)
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            logger.warning("Falling back to template workload")
            return self._generate_template_workload(iteration)
    
    def _validate_workload_format(self, code: str) -> bool:
        """Validate that workload follows required format"""
        required_patterns = [
            'def setup():',
            'def workload():',
            'timeit.repeat',
            'import timeit',
            'import statistics'
        ]
        
        for pattern in required_patterns:
            if pattern not in code:
                logger.warning(f"Missing required pattern: {pattern}")
                return False
        
        return True
    
    def _generate_template_workload(self, iteration: int) -> str:
        """Fallback template workload - FOLLOWS STANDARD FORMAT"""
        logger.info("Generating template workload code")
        
        templates = [
            # Template 1: Size variation
            """import timeit
import statistics

def setup():
    '''Setup test data with size variation'''
    global small_data, medium_data, large_data
    
    # Size variation tests
    small_data = list(range(100))
    medium_data = list(range(1000))
    large_data = list(range(10000))

def workload():
    '''Run workload'''
    global small_data, medium_data, large_data
    
    # Process each dataset
    # from main import process_data
    # result1 = process_data(small_data)
    # result2 = process_data(medium_data)
    # result3 = process_data(large_data)
    
    # Placeholder - replace with actual function calls
    _ = len(small_data) + len(medium_data) + len(large_data)

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
""",
            # Template 2: Pattern variation
            """import timeit
import statistics
import random

def setup():
    '''Setup test data with pattern variation'''
    global sequential, randomized, duplicates, sorted_data
    
    random.seed(42)
    
    # Pattern variation tests
    sequential = list(range(5000))
    randomized = random.sample(range(10000), 5000)
    duplicates = [random.randint(0, 100) for _ in range(5000)]
    sorted_data = sorted(randomized)

def workload():
    '''Run workload'''
    global sequential, randomized, duplicates, sorted_data
    
    # Process each pattern
    # from main import process_data
    # r1 = process_data(sequential)
    # r2 = process_data(randomized)
    # r3 = process_data(duplicates)
    # r4 = process_data(sorted_data)
    
    # Placeholder
    _ = len(sequential)

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
""",
            # Template 3: Edge cases
            """import timeit
import statistics

def setup():
    '''Setup edge case test data'''
    global empty, single, duplicates, reverse_sorted
    
    # Edge cases
    empty = []
    single = [42]
    duplicates = [1] * 1000
    reverse_sorted = list(range(5000, 0, -1))

def workload():
    '''Run workload'''
    global empty, single, duplicates, reverse_sorted
    
    # Test edge cases
    # from main import process_data
    # if empty: process_data(empty)
    # r1 = process_data(single)
    # r2 = process_data(duplicates)
    # r3 = process_data(reverse_sorted)
    
    # Placeholder
    _ = len(duplicates)

runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {statistics.mean(runtimes):.6f}")
print(f"Std Dev: {statistics.stdev(runtimes):.6f}")
"""
        ]
        
        template = templates[iteration % len(templates)]
        logger.warning("Using template workload - for testing only!")
        return template
    
    def _log_workload(self, iteration: int, workload_code: str) -> None:
        """Log generated workload code"""
        log_dir = Path("artifacts/workloads")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"workload_{iteration}.py"
        log_file.write_text(workload_code)
        logger.info(f"Workload code logged to {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Workload Generator Agent")
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--baseline-workload', required=True, help='Path to baseline workload file')
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--reference', help='Path to reference documentation')
    parser.add_argument('--prev-metrics', help='Path to previous metrics JSON (optional)')
    
    args = parser.parse_args()
    
    # Load baseline workload
    baseline_workload_code = Path(args.baseline_workload).read_text()
    
    # Load reference docs if provided
    reference_docs = None
    if args.reference and Path(args.reference).exists():
        reference_docs = Path(args.reference).read_text()
    
    # Load previous metrics if provided
    previous_metrics = None
    if args.prev_metrics and Path(args.prev_metrics).exists():
        with open(args.prev_metrics) as f:
            previous_metrics = json.load(f)
    
    # Note: In standalone mode, no MCP server
    # This is just for testing - normally called by orchestrator
    from mcp.server import MCPServer
    mcp = MCPServer()
    
    generator = WorkloadGenerator(mcp, model=args.model)
    workload_code = generator.generate(args.iteration, baseline_workload_code, 
                                       reference_docs, previous_metrics)
    
    # Output to stdout
    print(workload_code)