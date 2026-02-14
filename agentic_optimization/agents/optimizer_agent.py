"""
Optimizer Agent - Simplified Version

Key changes:
1. Only use Mini-SWE-Agent (no GPT fallback)
2. Skip file change detection entirely
3. Let orchestrator validate via workload testing
4. Return success marker when Mini-SWE completes
"""

import json
import logging
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict

# Import profiler types
try:
    from tools.profiler import ProfilingReport
except ImportError:
    ProfilingReport = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OptimizerAgent')


class CodeOptimizer:
    """Generates code optimizations using Mini-SWE-Agent"""
    
    def __init__(self, opencode_repo, mcp_server, model: str = "gpt-4o"):
        self.repo = opencode_repo
        self.mcp = mcp_server
        self.model = model
    
    def optimize(self, baseline_metrics: Dict, workload_type: str = "current",
                 workload_code: Optional[str] = None, retry_attempt: int = 1,
                 previous_patch_error: Optional[str] = None,
                 profiling_report = None) -> Optional[str]:
        """Generate optimization via Mini-SWE-Agent"""
        logger.info("Analyzing code for optimization opportunities...")
        logger.info(f"  Baseline execution time: {baseline_metrics.get('execution_time', 0):.3f}s")
        logger.info(f"  Memory: {baseline_metrics.get('memory_mb', 0):.1f}MB")
        logger.info(f"  Success rate: {baseline_metrics.get('success_rate', 0):.1%}")
        
        if profiling_report:
            logger.info(f"  📊 Profiling data available:")
            logger.info(f"     Functions profiled: {len(profiling_report.function_profiles)}")
            logger.info(f"     Hot lines: {len(profiling_report.line_profiles)}")
            logger.info(f"     Peak memory: {profiling_report.memory_profile.peak_mb:.1f} MB")
            logger.info(f"     Coverage: {profiling_report.coverage.overall_coverage_percent:.1f}%")
        
        if retry_attempt > 1:
            logger.info(f"  🔄 RETRY ATTEMPT #{retry_attempt} - trying different approach")
        
        logger.info("Running Mini-SWE-Agent optimization...")
        patch = self._optimize_with_mini_swe(
            baseline_metrics, workload_code, retry_attempt, profiling_report
        )
        
        if patch:
            self._log_patch(patch, retry_attempt)
            logger.info(f"✓ Mini-SWE-Agent completed")
        else:
            logger.warning("Mini-SWE-Agent failed")
        
        return patch

    def _optimize_with_mini_swe(self, baseline_metrics: Dict, 
                                workload_code: Optional[str] = None,
                                retry_attempt: int = 1,
                                profiling_report = None) -> Optional[str]:
        """Use Mini-SWE-Agent in yolo mode"""
        
        logger.info("Using Mini-SWE-Agent in yolo mode...")
        
        # Build workload context
        workload_info = ""
        if workload_code:
            workload_info = f"""
📝 WORKLOAD BEING TESTED:
```python
{workload_code}
```

Analyze this workload to understand:
- What operations are being measured
- What data patterns it uses
- What edge cases it tests
Optimize the code paths that this workload exercises.
"""
        
        # Add profiling context
        profiling_context = ""
        if profiling_report:
            profiling_context = f"""
🔬 PROFILING DATA - YOUR STARTING POINT 🔬

{profiling_report.to_llm_context()}

HOW TO USE THIS DATA:
✓ Hot functions show you where time is spent
✓ But the BEST optimization might not be IN those functions
✓ Consider:
  - Can you reduce CALLS to hot functions?
  - Are the DATA STRUCTURES used by hot functions optimal?
  - What do hot functions CALL? Can those be optimized?
  - Are there algorithmic improvements that span multiple functions?
"""
        
        retry_info = ""
        if retry_attempt > 1:
            retry_info = f"""
🔄 RETRY ATTEMPT #{retry_attempt} 🔄
Previous optimization attempts failed to meet the 5% improvement threshold.
Try a DIFFERENT optimization strategy:
- If you tried optimizing a hot function, try changing its data structures
- If you tried local optimizations, try algorithmic changes
- If you tried one approach, try a completely different one
Be more aggressive!
"""
        
        abs_repo_path = self.repo.root.absolute()
        
        task = f"""Optimize the library code in this repository for better performance.

🎯 TARGET: Achieve AT LEAST 5% improvement in execution time (CRITICAL for success)

📁 CODE LOCATION:
The library code you need to optimize is located at:
{abs_repo_path}

You MUST navigate to and edit files in this directory structure.
Use commands like:
  cd {abs_repo_path}
  find . -name "*.py" -type f | head -20
  ls -la

📊 Current Performance Metrics:
- Execution time: {baseline_metrics.get('execution_time', 0):.3f}s
- Memory usage: {baseline_metrics.get('memory_mb', 0):.1f}MB
- Success rate: {baseline_metrics.get('success_rate', 0):.1%}

{workload_info}{profiling_context}{retry_info}

🔍 YOUR OPTIMIZATION STRATEGY:

1. NAVIGATE TO THE CODE:
   cd {abs_repo_path}
   Explore the directory structure

2. START WITH PROFILING DATA (if available):
   - Hot functions are your PRIMARY targets
   - Trace call chains to find root causes
   - Consider data structures used across functions

3. UNDERSTAND THE WORKLOAD:
   - Analyze what operations are being tested
   - Identify code paths being exercised
   - Look for algorithmic improvements (O(n²) → O(n log n))

4. CONSIDER BROADER OPTIMIZATIONS:
   - Data structure changes (lists → sets, dicts → arrays)
   - Reduce memory allocations
   - Caching computed values
   - Batch processing
   - Lazy evaluation

5. DON'T BE AFRAID TO REFACTOR:
   - If profiling shows a hot function, look at:
     * What calls it (can we reduce calls?)
     * What it calls (are those efficient?)
     * The data structures it uses

📋 OPTIMIZATION CHECKLIST:
✓ Navigate to {abs_repo_path}
✓ Review profiling data (if available)
✓ Analyze the workload
✓ Find relevant source files
✓ Look for algorithmic improvements
✓ Consider data structure optimizations
✓ Reduce unnecessary allocations
✓ Test changes don't break correctness

⚠️ REQUIREMENTS:
- Maintain all public APIs
- Preserve correctness
- Edit files in {abs_repo_path}
- Add comments explaining optimizations

When done optimizing, run:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"""
        
        try:
            output_dir = Path.cwd() / 'artifacts' / 'mini_swe_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            traj_file = output_dir / f"trajectory_retry{retry_attempt}.json"
            
            import shlex
            task_arg = f"$(cat <<'EOF'\n{task}\nEOF\n)"
            cmd_str = f"mini -y -m {shlex.quote(self.model)} -t {shlex.quote(task_arg)} -o {shlex.quote(str(traj_file.absolute()))} -c mini.yaml"
            
            logger.info(f"Running mini-SWE-agent...")
            logger.info(f"Task length: {len(task)} chars")
            logger.info(f"Target repo: {abs_repo_path}")
            
            result = subprocess.run(
                f"yes '' | {cmd_str}",
                shell=True,
                cwd=self.repo.root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                logger.error(f"Mini-SWE-Agent failed with code {result.returncode}")
                return None
            
            logger.info("Mini-SWE-Agent completed")
            
            # Check if trajectory exists
            if traj_file.exists() and traj_file.stat().st_size > 0:
                logger.info(f"✓ Mini-SWE-Agent completed (trajectory: {traj_file})")
                
                # Return success marker - let orchestrator validate via workload
                return f"# Mini-SWE-Agent completed\n# See trajectory: {traj_file}"
            
            logger.warning("No trajectory file generated")
            return None
                
        except subprocess.TimeoutExpired:
            logger.error("Mini-SWE-Agent timed out after 10 minutes")
            return None
        except FileNotFoundError:
            logger.warning("mini command not found. Install: pip install mini-swe-agent")
            return None
        except Exception as e:
            logger.error(f"Mini-SWE-Agent error: {e}", exc_info=True)
            return None
    
    def _log_patch(self, patch: str, retry_attempt: int = 1) -> None:
        """Log optimization patch"""
        log_dir = Path("artifacts/optimizations")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        existing = list(log_dir.glob("optimization_*.diff"))
        patch_id = len(existing) + 1
        
        retry_suffix = f"_retry{retry_attempt}" if retry_attempt > 1 else ""
        log_file = log_dir / f"optimization_{patch_id}{retry_suffix}.diff"
        log_file.write_text(patch)
        logger.info(f"Optimization patch logged to {log_file}")


if __name__ == "__main__":
    import argparse
    from opencode.repo import OpenCodeRepo
    from mcp.server import MCPServer
    
    parser = argparse.ArgumentParser(description="Optimizer Agent")
    parser.add_argument('--repo', required=True, help='Repository path')
    parser.add_argument('--metrics', required=True, help='Metrics JSON file')
    parser.add_argument('--workload', help='Workload code file')
    parser.add_argument('--retry', type=int, default=1, help='Retry attempt number')
    parser.add_argument('--model', default='gpt-4o')
    
    args = parser.parse_args()
    
    with open(args.metrics) as f:
        metrics = json.load(f)
    
    workload_code = None
    if args.workload and Path(args.workload).exists():
        workload_code = Path(args.workload).read_text()
    
    repo = OpenCodeRepo(args.repo, mode="direct")
    mcp = MCPServer()
    
    optimizer = CodeOptimizer(repo, mcp, model=args.model)
    patch = optimizer.optimize(metrics, workload_code=workload_code, retry_attempt=args.retry)
    
    if patch:
        print(patch)
    else:
        logger.error("Failed to generate optimization")
        sys.exit(1)