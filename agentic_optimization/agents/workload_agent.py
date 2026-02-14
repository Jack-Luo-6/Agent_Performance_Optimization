"""
Diverse Workload Generator - ConcoLLMic-Inspired Strategy
Generates diverse performance test workloads using systematic path exploration

Strategy (inspired by ConcoLLMic):
1. Static Analysis: Extract CFG, identify unexplored/deep paths
2. Heuristic Scoring: Score paths by complexity (loop depth, recursion, path length)
3. Diverse Generation: Create workloads targeting DIFFERENT patterns than baseline
4. Iterative Feedback: Use profiling to avoid redundancy

Agent autonomously decides:
- How many workloads to generate (based on code complexity)
- What each workload targets (different path patterns)
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DiverseWorkloadGenerator')


@dataclass
class WorkloadInfo:
    """Information about a workload"""
    name: str
    code: str
    description: str
    is_original: bool = False


class DiverseWorkloadGenerator:
    """
    Generates diverse workloads using ConcoLLMic-inspired strategy
    
    Key principles:
    1. AVOID baseline patterns (use profiling to see what's already covered)
    2. TARGET deep/complex paths (use heuristic scoring)
    3. DIVERSIFY systematically (each workload targets different pattern)
    4. LET AGENT DECIDE quantity (based on code complexity)
    """
    
    def __init__(self, mcp_server, model: str = "gpt-4o"):
        self.mcp = mcp_server
        self.model = model
    
    def generate_diverse_workloads(self,
                                   baseline_workload_code: str,
                                   target_repo_path: Path,
                                   profiling_report = None,
                                   baseline_metrics = None) -> List[WorkloadInfo]:
        """
        Generate diverse workloads
        
        Args:
            baseline_workload_code: Original workload (to AVOID its patterns)
            target_repo_path: Path to target code
            profiling_report: Profiling data showing current coverage
            baseline_metrics: Baseline performance metrics
        
        Returns:
            List of WorkloadInfo objects (agent decides how many)
        """
        logger.info("Generating diverse workloads using ConcoLLMic-inspired strategy...")
        
        # Build context for agent
        profiling_context = self._build_profiling_context(profiling_report)
        baseline_context = self._build_baseline_context(baseline_workload_code, baseline_metrics)
        strategy_context = self._build_strategy_context()
        
        # Call Mini-SWE-Agent to generate workloads
        workload_infos = self._generate_with_mini_swe(
            baseline_workload_code=baseline_workload_code,
            target_repo_path=target_repo_path,
            profiling_context=profiling_context,
            baseline_context=baseline_context,
            strategy_context=strategy_context
        )
        
        if not workload_infos:
            logger.warning("Failed to generate diverse workloads")
            return []
        
        logger.info(f"✓ Generated {len(workload_infos)} diverse workload(s)")
        
        return workload_infos
    
    def _build_profiling_context(self, profiling_report) -> str:
        """Build context showing what's already covered (to AVOID)"""
        if not profiling_report:
            return """
🔬 PROFILING DATA: Not available

Without profiling data, focus on generating diverse algorithmic patterns:
- Different data structures (sorted, random, pathological)
- Different operation patterns (sequential, random access, worst-case)
- Different edge cases (empty, single element, duplicates, extremes)
"""
        
        # Extract coverage gaps
        uncovered_info = ""
        if profiling_report.coverage.uncovered_lines:
            total_uncovered = sum(len(lines) for lines in profiling_report.coverage.uncovered_lines.values())
            uncovered_info = f"""
📊 COVERAGE GAPS (TARGET THESE):
- Total uncovered lines: {total_uncovered}
- Files with gaps: {len(profiling_report.coverage.uncovered_lines)}
- Overall coverage: {profiling_report.coverage.overall_coverage_percent:.1f}%

Your workloads should try to execute these uncovered paths!
"""
        
        # Extract hot paths (to DIVERSIFY FROM)
        hot_paths_info = ""
        if profiling_report.function_profiles:
            covered_funcs = [fp.name for fp in profiling_report.function_profiles[:5]]
            hot_paths_info = f"""
🔥 ALREADY COVERED (DIVERSIFY FROM THESE):
Top functions executed by baseline:
{chr(10).join(f'  • {func}' for func in covered_funcs)}

Your workloads should:
- Call DIFFERENT functions (explore other code paths)
- Use DIFFERENT data patterns (avoid similar inputs)
- Target UNCOVERED branches
"""
        
        return f"""
🔬 PROFILING DATA - GUIDE YOUR DIVERSITY STRATEGY

{uncovered_info}

{hot_paths_info}

{profiling_report.to_llm_context() if profiling_report else ''}
"""
    
    def _build_baseline_context(self, baseline_code: str, baseline_metrics) -> str:
        """Build context showing baseline workload (to DIFFERENTIATE FROM)"""
        metrics_info = ""
        if baseline_metrics:
            metrics_info = f"""
📊 Baseline Performance:
- Execution time: {baseline_metrics.execution_time:.3f}s
- Memory: {baseline_metrics.memory_mb:.1f}MB
- Success rate: {baseline_metrics.success_rate:.1%}
"""
        
        return f"""
📝 BASELINE WORKLOAD - ANALYZE TO DIFFERENTIATE

{metrics_info}

```python
{baseline_code}
```

CRITICAL ANALYSIS QUESTIONS:
1. What functions does baseline call? → Target DIFFERENT functions
2. What data patterns does it use? → Use DIFFERENT patterns
3. What operations does it perform? → Test DIFFERENT operations
4. What edge cases does it miss? → Target those edge cases

Your diverse workloads MUST:
✓ Test the SAME codebase (same imports/modules)
✓ Use DIFFERENT code paths than baseline
✓ Create DIFFERENT stress patterns
✗ DO NOT just scale up baseline (that's not diversity!)
"""
    
    def _build_strategy_context(self) -> str:
        """Build ConcoLLMic-inspired strategy context"""
        return """
🎯 CONCOLLMIC-INSPIRED DIVERSITY STRATEGY

PRINCIPLE: Systematic exploration of DIVERSE execution paths

1️⃣ STATIC ANALYSIS (your job):
   - Identify code structure (loops, recursion, branches)
   - Find deep/complex paths (high cyclomatic complexity)
   - Locate uncovered regions (from profiling data)

2️⃣ HEURISTIC SCORING (prioritize):
   - Loop nesting depth × 3 (nested loops = high priority)
   - Recursion depth × 5 (recursive calls = high priority)
   - Path length × 2 (deep call chains = high priority)
   - Cyclomatic complexity × 1 (many branches = high priority)

3️⃣ DIVERSITY PATTERNS (each workload targets ONE):
   Pattern A: ALGORITHMIC WORST-CASE
     - Pathological inputs (reverse sorted for sort, collisions for hash)
     - Trigger O(n²) paths in O(n log n) algorithms
   
   Pattern B: DEEP RECURSION/NESTING
     - Inputs causing maximum recursion depth
     - Deeply nested data structures
   
   Pattern C: EDGE CASE COMBINATIONS
     - Empty + boundary + extreme values together
     - Unusual type combinations
   
   Pattern D: MEMORY PRESSURE
     - Large data structures
     - Many allocations/deallocations
   
   Pattern E: UNCOVERED BRANCHES
     - Target specific uncovered lines from profiling
     - Error paths, exceptional cases

4️⃣ AGENT DECISION (YOU DECIDE):
   Based on code complexity, decide HOW MANY workloads:
   - Simple code (< 500 LoC, few branches): 2-3 workloads
   - Medium code (500-2000 LoC, moderate complexity): 3-5 workloads
   - Complex code (> 2000 LoC, high complexity): 5-8 workloads
   
   Each workload should target a DIFFERENT pattern above!

5️⃣ IMPLEMENTATION RULES:
   ✓ Each workload = 1 Python file with timeit benchmark
   ✓ Same format as baseline (setup(), workload(), timeit.repeat)
   ✓ DIFFERENT code paths than baseline
   ✓ Clear description of what pattern it targets
   ✗ DO NOT just copy baseline with bigger numbers!
"""
    
    def _generate_with_mini_swe(self,
                                baseline_workload_code: str,
                                target_repo_path: Path,
                                profiling_context: str,
                                baseline_context: str,
                                strategy_context: str) -> List[WorkloadInfo]:
        """Use Mini-SWE-Agent to generate diverse workloads"""
        
        abs_repo_path = target_repo_path.absolute()
        output_dir = Path.cwd() / 'artifacts' / 'diverse_workloads'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"""Generate DIVERSE performance test workloads for systematic path exploration.

🎯 YOUR MISSION: Create multiple DIFFERENT workloads that stress DIFFERENT code paths

📁 TARGET CODE LOCATION:
{abs_repo_path}

First, explore the code structure:
cd {abs_repo_path}
find . -name "*.py" -type f | grep -v __pycache__ | head -20
ls -la

{baseline_context}

{profiling_context}

{strategy_context}

📋 STEP-BY-STEP PROCESS:

STEP 1: ANALYZE CODE STRUCTURE
- Explore {abs_repo_path}
- Identify main modules and functions
- Estimate code complexity (LoC, branches, loops)
- Review profiling data for coverage gaps

STEP 2: DECIDE QUANTITY
Based on complexity, decide how many diverse workloads to create (2-8).
You MUST create a plan file first:

cat > {output_dir}/diversity_plan.json << 'EOF'
{{
  "num_workloads": <YOUR_DECISION>,
  "reasoning": "<why this many workloads>",
  "workloads": [
    {{
      "name": "diverse_1",
      "pattern": "<which pattern: worst_case/deep_recursion/edge_cases/memory/uncovered>",
      "target": "<what specific code path or function>",
      "differentiation": "<how it differs from baseline>"
    }},
    ...
  ]
}}
EOF

STEP 3: GENERATE EACH WORKLOAD
For each workload in your plan, create a file:

{output_dir}/diverse_<N>.py

Each file MUST follow this EXACT format:

```python
import timeit
import statistics
# Add required imports (MATCH baseline imports!)

def setup():
    '''Setup - create test data targeting <PATTERN>'''
    global test_data  # Declare globals
    
    # Create data for THIS pattern
    # Example for worst_case: reverse sorted for sort algorithm
    # Example for deep_recursion: deeply nested structure
    test_data = ...

def workload():
    '''Workload - execute target functions on test data'''
    global test_data
    
    # Call SAME functions as baseline
    # but with DIFFERENT data patterns
    # from target_module import target_function
    # result = target_function(test_data)
    pass

# Benchmark (DO NOT MODIFY)
runtimes = timeit.repeat(workload, number=1, repeat=3, setup=setup)
print(f"Mean: {{statistics.mean(runtimes):.6f}}")
print(f"Std Dev: {{statistics.stdev(runtimes):.6f}}")
```

STEP 4: CREATE METADATA FILE
After generating all workloads:

cat > {output_dir}/workloads_manifest.json << 'EOF'
[
  {{
    "name": "diverse_1",
    "description": "<full description of what this tests>",
    "file": "diverse_1.py"
  }},
  ...
]
EOF

⚠️ CRITICAL REQUIREMENTS:
1. CREATE diversity_plan.json FIRST (shows your reasoning)
2. Each workload file MUST be valid Python
3. Each workload MUST target DIFFERENT pattern
4. All workloads MUST test SAME functions/modules as baseline
5. Each workload MUST have clear differentiation from baseline
6. CREATE workloads_manifest.json LAST (I will parse this!)

📁 FILES TO CREATE:
{output_dir}/diversity_plan.json        (your analysis & plan)
{output_dir}/diverse_1.py               (workload 1)
{output_dir}/diverse_2.py               (workload 2)
...
{output_dir}/workloads_manifest.json    (final metadata)

VERIFY YOUR WORK:
ls -la {output_dir}/
cat {output_dir}/workloads_manifest.json

When done, run:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"""
        
        try:
            traj_dir = Path.cwd() / 'artifacts' / 'mini_swe_output'
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_file = traj_dir / "diverse_workload_gen.json"
            
            import shlex
            task_arg = f"$(cat <<'EOF'\n{task}\nEOF\n)"
            cmd_str = f"mini -y -m {shlex.quote(self.model)} -t {shlex.quote(task_arg)} -o {shlex.quote(str(traj_file.absolute()))} -c mini.yaml"
            
            logger.info(f"Running Mini-SWE-Agent for diverse workload generation...")
            logger.info(f"Task length: {len(task)} chars")
            logger.info(f"Output directory: {output_dir}")
            
            result = subprocess.run(
                f"yes '' | {cmd_str}",
                shell=True,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes for multiple workloads
            )
            
            if result.returncode != 0:
                logger.error(f"Mini-SWE-Agent failed with code {result.returncode}")
                return []
            
            logger.info("Mini-SWE-Agent completed")
            
            # Parse manifest
            manifest_file = output_dir / "workloads_manifest.json"
            
            if not manifest_file.exists():
                logger.warning("⚠️  Manifest file not created")
                # Try to discover workload files
                return self._discover_workloads(output_dir)
            
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            workload_infos = []
            for entry in manifest:
                workload_file = output_dir / entry['file']
                if not workload_file.exists():
                    logger.warning(f"⚠️  Workload file not found: {workload_file}")
                    continue
                
                workload_code = workload_file.read_text()
                
                # Validate format
                if not self._validate_workload_format(workload_code):
                    logger.warning(f"⚠️  Invalid workload format: {entry['name']}")
                    continue
                
                workload_infos.append(WorkloadInfo(
                    name=entry['name'],
                    code=workload_code,
                    description=entry['description'],
                    is_original=False
                ))
            
            return workload_infos
                
        except subprocess.TimeoutExpired:
            logger.error("Mini-SWE-Agent timed out after 15 minutes")
            return []
        except FileNotFoundError:
            logger.warning("mini command not found. Install: pip install mini-swe-agent")
            return []
        except Exception as e:
            logger.error(f"Mini-SWE-Agent error: {e}", exc_info=True)
            return []
    
    def _discover_workloads(self, output_dir: Path) -> List[WorkloadInfo]:
        """Fallback: discover workload files without manifest"""
        logger.info("Attempting to discover workload files...")
        
        workload_files = sorted(output_dir.glob("diverse_*.py"))
        
        if not workload_files:
            logger.warning("No workload files found")
            return []
        
        workload_infos = []
        for idx, wl_file in enumerate(workload_files, 1):
            workload_code = wl_file.read_text()
            
            if not self._validate_workload_format(workload_code):
                logger.warning(f"⚠️  Invalid format: {wl_file.name}")
                continue
            
            workload_infos.append(WorkloadInfo(
                name=f"diverse_{idx}",
                code=workload_code,
                description=f"Discovered from {wl_file.name}",
                is_original=False
            ))
        
        logger.info(f"✓ Discovered {len(workload_infos)} workload(s)")
        return workload_infos
    
    def _validate_workload_format(self, code: str) -> bool:
        """Validate workload follows required format"""
        required_patterns = [
            'def setup():',
            'def workload():',
            'timeit.repeat',
            'import timeit',
            'import statistics'
        ]
        
        for pattern in required_patterns:
            if pattern not in code:
                return False
        
        return True


if __name__ == "__main__":
    import argparse
    from mcp.server import MCPServer
    
    parser = argparse.ArgumentParser(description="Diverse Workload Generator")
    parser.add_argument('--baseline-workload', required=True)
    parser.add_argument('--target-repo', required=True)
    parser.add_argument('--model', default='gpt-4o')
    
    args = parser.parse_args()
    
    baseline_workload_code = Path(args.baseline_workload).read_text()
    
    mcp = MCPServer()
    generator = DiverseWorkloadGenerator(mcp, model=args.model)
    
    workload_infos = generator.generate_diverse_workloads(
        baseline_workload_code=baseline_workload_code,
        target_repo_path=Path(args.target_repo),
        profiling_report=None,
        baseline_metrics=None
    )
    
    for wl in workload_infos:
        print(f"\n{'='*80}")
        print(f"Workload: {wl.name}")
        print(f"Description: {wl.description}")
        print(f"{'='*80}")
        print(wl.code)