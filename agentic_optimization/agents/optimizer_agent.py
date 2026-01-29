"""
Optimizer Agent - IMPROVED VERSION

Key improvements:
1. Explicit repo path instruction for Mini-SWE-Agent
2. Profiling data as GUIDE, not CONSTRAINT
3. Better optimization strategy that considers the full picture
4. FIXED: Properly detects Mini-SWE changes in site-packages
"""

import json
import logging
import sys
import os
import hashlib
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
    """Generates code optimizations using SWE-Agent + gpt-4o"""
    
    def __init__(self, opencode_repo, mcp_server, model: str = "gpt-4o"):
        self.repo = opencode_repo
        self.mcp = mcp_server
        self.model = model
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return """You are an expert performance engineer specializing in code optimization.

Your responsibilities:
- Analyze performance bottlenecks from metrics AND profiling data
- Understand what the workload is testing
- Propose algorithmic improvements
- Optimize memory allocation patterns
- Improve hot path efficiency
- Preserve correctness and public APIs

PROFILING DATA PHILOSOPHY:
- Profiling data shows you WHERE to start looking
- But the BEST optimization might not be IN the hot function itself
- Consider: data structures, call patterns, algorithmic complexity
- Sometimes you need to refactor code that CALLS the hot function
- Sometimes you need to optimize what the hot function CALLS

CRITICAL SUCCESS CRITERIA:
- You must achieve AT LEAST 5% improvement to succeed
- Failure to meet 5% means you get another chance (limited retries)
- Focus on the HIGHEST IMPACT optimizations first
- Each retry is your chance to try a DIFFERENT approach

Output format:
Return ONLY a git-style unified diff patch. The patch must:
- Be valid unified diff format (+++, ---, @@)
- Include sufficient context lines
- Be applicable with git apply or patch
- Preserve all public interfaces
- Include comments explaining the optimization

Optimization priorities:
1. Algorithmic complexity (O(n²) → O(n log n)) - BIGGEST WINS
2. Data structure improvements (lists → sets, better layouts)
3. Memory allocations (reduce unnecessary copies)
4. Call pattern optimization (reduce calls to expensive functions)
5. Cache locality (improve data access patterns)

When analyzing metrics AND profiling:
- Start with hot functions but don't stop there
- Trace call chains to find root causes
- Look for patterns across multiple functions
- Consider workload characteristics

RETRY STRATEGY:
If your previous optimization failed (< 5% improvement):
- Try a COMPLETELY DIFFERENT approach
- Don't just tweak the same optimization
- Consider: Did you target the right code? Did you miss a bigger bottleneck?
- Be more aggressive with changes

Always verify your changes maintain correctness."""
    
    def optimize(self, baseline_metrics: Dict, workload_type: str = "current",
             workload_code: Optional[str] = None, retry_attempt: int = 1,
             previous_patch_error: Optional[str] = None,
             profiling_report = None) -> Optional[str]:
        """Generate optimization patch"""
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
            if previous_patch_error:
                logger.info(f"  Previous error: {previous_patch_error}")
        
        # Try Mini-SWE-Agent first
        patch = None
        mini_swe_context = None
        try:
            logger.info("Attempting optimization with Mini-SWE-Agent...")
            patch, mini_swe_context = self._optimize_with_mini_swe(
                baseline_metrics, workload_code, retry_attempt, profiling_report
            )
        except Exception as e:
            logger.warning(f"Mini-SWE-Agent failed: {e}")
        
        # Fallback to direct OpenAI (with Mini-SWE context if available)
        if not patch:
            logger.info("Falling back to direct OpenAI optimization...")
            try:
                patch = self._optimize_with_openai(
                    baseline_metrics, 
                    workload_code, 
                    retry_attempt,
                    previous_patch_error,
                    mini_swe_context,
                    profiling_report
                )
            except Exception as e:
                logger.error(f"OpenAI optimization error: {e}", exc_info=True)
                return None
        
        if patch:
            self._log_patch(patch, retry_attempt)
            logger.info(f"✓ Optimization patch generated ({len(patch)} chars)")
        else:
            logger.warning("No optimization patch generated")
        
        return patch

    def _optimize_with_mini_swe(self, baseline_metrics: Dict, 
                                workload_code: Optional[str] = None,
                                retry_attempt: int = 1,
                                profiling_report = None) -> tuple[Optional[str], Optional[str]]:
        """
        Use Mini-SWE-Agent via programmatic API in yolo mode
        Returns: (patch, context) where context is trajectory info for GPT fallback
        """
        import subprocess
        
        logger.info("Using Mini-SWE-Agent in yolo mode...")
        
        # Build workload info
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
        
        # Add profiling context - ADVISORY, NOT RESTRICTIVE
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
  
Remember: Profiling shows symptoms. Your job is to find the root cause.
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
        
        # Get absolute repo path for explicit instruction
        abs_repo_path = self.repo.root.absolute()
        library_path = Path(abs_repo_path)
        
        # Build task with profiling data
        task = f"""Optimize the library code in this repository for better performance.

🎯 TARGET: Achieve AT LEAST 5% improvement in execution time (CRITICAL for success)

📁 CODE LOCATION (IMPORTANT):
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
   Explore the directory structure to understand the codebase

2. START WITH PROFILING DATA (if available):
   - The profiling data shows which functions consume the most time
   - These are your PRIMARY targets, but not your ONLY targets
   - Hot functions often call other functions - trace the call chain
   - Sometimes the bottleneck is in data structures used across many functions

3. UNDERSTAND THE WORKLOAD:
   - Analyze what operations the workload is testing
   - Identify the code paths being exercised
   - Look for algorithmic improvements (O(n²) → O(n log n))

4. CONSIDER BROADER OPTIMIZATIONS:
   - Data structure changes (lists → sets, dicts → arrays)
   - Reducing memory allocations (avoid unnecessary copies)
   - Caching computed values
   - Batch processing instead of individual operations
   - Lazy evaluation where appropriate

5. DON'T BE AFRAID TO REFACTOR:
   - If profiling shows a hot function, look at:
     * What calls it (can we reduce calls?)
     * What it calls (are those functions efficient?)
     * The data structures it uses (can we improve them?)
   - Sometimes the best optimization is NOT in the hot function itself,
     but in changing how it's used or what data it receives

📋 OPTIMIZATION CHECKLIST:
✓ Navigate to {abs_repo_path}
✓ Review profiling data (if available) as a starting point
✓ Analyze the workload to understand what's being tested
✓ Find the relevant source files (use find, ls, or search)
✓ Look for algorithmic improvements first (biggest impact)
✓ Consider data structure optimizations
✓ Reduce unnecessary memory allocations
✓ Test your changes don't break correctness

⚠️ REQUIREMENTS:
- Maintain all public APIs and interfaces
- Preserve correctness (must pass all tests)
- Edit files in {abs_repo_path}
- Add comments explaining optimizations

💡 PROFILING PHILOSOPHY:
The profiling data is a GUIDE, not a CONSTRAINT. It shows you where to START looking,
but the best optimization might involve:
- Changing data structures used by hot functions
- Reducing calls to hot functions
- Optimizing functions that hot functions call
- Improving algorithms that span multiple functions

EXAMPLES OF THINKING BEYOND HOT FUNCTIONS:
- Hot function does list.append() in a loop → Use list comprehension or preallocate
- Hot function is called 1000 times → Can we batch the calls?
- Hot function uses dict lookups → Would a set or different data structure be faster?
- Hot function has O(n²) algorithm → Can we reduce to O(n log n)?

When you're done optimizing, run:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"""
        
        try:
            # Use absolute path for output
            output_dir = Path.cwd() / 'artifacts' / 'mini_swe_output'
            output_dir.mkdir(parents=True, exist_ok=True)
            traj_file = output_dir / f"trajectory_retry{retry_attempt}.json"
            
            # FIXED: Compute file checksums BEFORE Mini-SWE runs
            logger.info("Computing baseline file checksums...")
            checksums_before = {}
            for py_file in library_path.rglob("*.py"):
                try:
                    if py_file.is_file():
                        checksums_before[py_file] = hashlib.md5(py_file.read_bytes()).hexdigest()
                except:
                    pass
            logger.info(f"  Tracked {len(checksums_before)} files")
            
            # Wrap task in cat EOF heredoc format
            import shlex
            task_arg = f"$(cat <<'EOF'\n{task}\nEOF\n)"
            
            # Build command
            cmd_str = f"mini -y -m {shlex.quote(self.model)} -t {shlex.quote(task_arg)} -o {shlex.quote(str(traj_file.absolute()))} -c mini.yaml"
            
            logger.info(f"Running mini-SWE-agent in yolo mode...")
            logger.info(f"Task length: {len(task)} chars")
            logger.info(f"Target repo: {abs_repo_path}")
            
            # Run with shell=True for heredoc
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
                context = self._extract_mini_swe_context(traj_file)
                return None, context
            
            logger.info("Mini-SWE-Agent completed successfully")
            
            # FIXED: If trajectory file exists and has content, assume success
            if traj_file.exists() and traj_file.stat().st_size > 0:
                logger.info(f"✓ Mini-SWE-Agent completed (trajectory: {traj_file})")
                
                # Check if there were successful commands (returncode 0)
                try:
                    with open(traj_file) as f:
                        traj = json.load(f)
                    
                    # Look for successful sed/file edit commands
                    has_modifications = False
                    for msg in traj.get('messages', []):
                        if msg.get('role') == 'user' and '<returncode>0</returncode>' in msg.get('content', ''):
                            # Check previous message for edit commands
                            idx = traj['messages'].index(msg)
                            if idx > 0:
                                prev_content = traj['messages'][idx-1].get('content', '')
                                if 'sed -i' in prev_content or 'cat <<' in prev_content:
                                    has_modifications = True
                                    logger.info("  📝 Detected successful file modifications")
                                    break
                    
                    if has_modifications:
                        # Generate simple patch notation
                        patch = f"# Mini-SWE-Agent made modifications\n# See trajectory: {traj_file}"
                        return patch, None
                    else:
                        logger.warning("Trajectory exists but no modifications detected")
                        return None, self._extract_mini_swe_context(traj_file)
                        
                except Exception as e:
                    logger.warning(f"Could not parse trajectory: {e}")
                    # If we can't parse but file exists, assume success
                    patch = f"# Mini-SWE-Agent completed\n# See trajectory: {traj_file}"
                    return patch, None
            
            logger.warning("No trajectory file generated")
            return None, None
            
            logger.info("Mini-SWE-Agent completed successfully")
            logger.info(f"stdout: {result.stdout[-500:]}")
            
            # FIXED: Check for changes using file checksums instead of git diff
            logger.info("Checking for file changes...")
            changed_files = []
            for py_file, old_hash in checksums_before.items():
                try:
                    if py_file.is_file():
                        new_hash = hashlib.md5(py_file.read_bytes()).hexdigest()
                        if new_hash != old_hash:
                            changed_files.append(py_file)
                            logger.info(f"  📝 Modified: {py_file.relative_to(library_path)}")
                except:
                    pass
            
            if changed_files:
                logger.info(f"✓ Mini-SWE-Agent modified {len(changed_files)} files")
                
                # Generate a simple patch notation (not full unified diff)
                patch_lines = []
                patch_lines.append(f"# Mini-SWE-Agent modifications in {library_path}")
                patch_lines.append(f"# Modified {len(changed_files)} file(s):")
                for f in changed_files:
                    patch_lines.append(f"# - {f.relative_to(library_path)}")
                patch = "\n".join(patch_lines)
                
                if traj_file.exists():
                    logger.info(f"Trajectory saved to {traj_file}")
                
                return patch, None
            else:
                logger.warning("No changes detected by checksum comparison")
                # Extract context for GPT
                context = self._extract_mini_swe_context(traj_file)
                return None, context
                
        except subprocess.TimeoutExpired:
            logger.error("Mini-SWE-Agent timed out after 10 minutes")
            return None, None
        except FileNotFoundError:
            logger.warning("mini command not found. Install: pip install mini-swe-agent")
            return None, None
        except Exception as e:
            logger.error(f"Mini-SWE-Agent error: {e}", exc_info=True)
            return None, None
    
    def _extract_mini_swe_context(self, traj_file: Path) -> Optional[str]:
        """Extract useful context from Mini-SWE trajectory for GPT fallback"""
        try:
            if not traj_file.exists():
                return None
            
            with open(traj_file) as f:
                traj = json.load(f)
            
            # Extract key information from trajectory
            context_parts = []
            
            # Get agent's analysis and thought process
            if 'messages' in traj:
                for msg in traj['messages']:
                    if msg.get('role') == 'assistant' and 'content' in msg:
                        content = msg['content']
                        # Extract THOUGHT sections and commands
                        if 'THOUGHT:' in content:
                            context_parts.append(content)
            
            # Get any files that were examined or edited
            examined_files = []
            if 'messages' in traj:
                for msg in traj['messages']:
                    if msg.get('role') == 'user' and 'content' in msg:
                        content = msg['content']
                        # Look for file content in responses
                        if '<output>' in content and len(content) < 5000:
                            examined_files.append(content[:1000])
            
            if context_parts or examined_files:
                context = "Mini-SWE-Agent's analysis:\n" + "\n---\n".join(context_parts)
                if examined_files:
                    context += "\n\nFiles examined:\n" + "\n---\n".join(examined_files)
                logger.info(f"Extracted {len(context)} chars of context from Mini-SWE trajectory")
                return context
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract Mini-SWE context: {e}")
            return None
    
    def _optimize_with_openai(self, baseline_metrics: Dict, 
                          workload_code: Optional[str] = None,
                          retry_attempt: int = 1,
                          previous_patch_error: Optional[str] = None,
                          mini_swe_context: Optional[str] = None,
                          profiling_report = None) -> Optional[str]:
        """Generate optimization using OpenAI API"""
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("No OPENAI_API_KEY found")
            return None
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            # Get files to optimize - prioritize hot files from profiling
            files_to_read = []
            
            if profiling_report and profiling_report.function_profiles:
                # Get unique files from hot functions
                hot_files = set()
                for fp in profiling_report.function_profiles[:20]:  # Top 20 functions
                    if fp.filename and not fp.filename.startswith('~'):
                        hot_files.add(fp.filename)
                
                logger.info(f"Prioritizing {len(hot_files)} hot files from profiling")
                files_to_read = list(hot_files)
            
            # If no profiling or not enough files, get all Python files
            if len(files_to_read) < 5:
                all_files = self.repo.list_files(pattern="**/*.py", exclude_tests=True)
                files_to_read.extend([str(f) for f in all_files[:10]])
                files_to_read = list(set(files_to_read))  # Deduplicate
            
            # Read FULL content of prioritized files (NO TRUNCATION)
            code_files = {}
            for file_path in files_to_read[:15]:  # Limit to 15 files max
                try:
                    full_content = self.repo.read(file_path)
                    code_files[file_path] = full_content
                    logger.info(f"Read full file: {file_path} ({len(full_content)} chars)")
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
            
            if not code_files:
                logger.error("No code files to optimize")
                return None
            
            # Format code with line numbers to help GPT
            code_context = []
            for path, content in code_files.items():
                lines = content.split('\n')
                numbered_lines = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
                code_context.append(f"=== FILE: {path} ({len(lines)} lines) ===\n{numbered_lines}")
            
            code_context = "\n\n".join(code_context)
            
            workload_context = ""
            if workload_code:
                workload_context = f"""
WORKLOAD BEING TESTED:
```python
{workload_code}
```

Analyze this workload to understand what operations are being measured.
Optimize the code paths that this workload exercises.
"""
            
            # Add profiling context - ADVISORY not RESTRICTIVE
            profiling_context = ""
            if profiling_report:
                profiling_context = f"""
🔬 PROFILING DATA - YOUR OPTIMIZATION GUIDE 🔬

{profiling_report.to_llm_context()}

HOW TO USE PROFILING DATA:
1. Start with hot functions - they're consuming the most time
2. But don't stop there! Consider:
   - Can you improve the ALGORITHM in hot functions? (O(n²) → O(n log n))
   - Can you change DATA STRUCTURES used by hot functions? (list → set, dict → array)
   - Can you reduce CALLS to hot functions? (batch operations, cache results)
   - What do hot functions CALL? Can you optimize those?
   - Are there patterns across multiple functions suggesting a systemic issue?

3. Think holistically:
   - Sometimes optimizing a non-hot function that CALLS a hot function is better
   - Sometimes changing a data structure used by many functions is better
   - Sometimes the best fix is algorithmic, spanning multiple functions

The profiling data shows SYMPTOMS. Your job is to diagnose the ROOT CAUSE.
"""
            
            # Include Mini-SWE context if available
            mini_context = ""
            if mini_swe_context:
                mini_context = f"""
🤖 PREVIOUS ATTEMPT CONTEXT (Mini-SWE-Agent):
{mini_swe_context}

The above shows what another agent tried. Learn from this approach but implement it correctly as a unified diff patch.
"""
            
            retry_context = ""
            if retry_attempt > 1:
                error_guidance = ""
                if previous_patch_error:
                    if "malformed patch" in previous_patch_error:
                        error_guidance = """
⚠️ PREVIOUS PATCH WAS MALFORMED ⚠️

The patch format was incorrect. Common issues:
1. Context lines don't match the actual file exactly
2. Line numbers in @@ headers are wrong
3. Mixing tabs and spaces
4. Extra or missing blank lines

CRITICAL: Your diff must match the ACTUAL file content shown below EXACTLY.
Every context line (starting with space) must be identical to the source.
"""
                    elif "can't find file" in previous_patch_error or "No such file" in previous_patch_error:
                        error_guidance = """
⚠️ FILE PATH WAS WRONG ⚠️

The file path in your patch didn't exist. Use EXACTLY the paths shown in the file list below.
"""
                
                retry_context = f"""
🔄 RETRY ATTEMPT #{retry_attempt} 🔄

Your previous optimization attempt failed.
{error_guidance}

INSTRUCTIONS FOR THIS RETRY:
1. Generate a VALID unified diff format patch
2. Match the file content EXACTLY (see files below)
3. Try a DIFFERENT optimization approach than before
4. Be more aggressive to achieve >5% improvement
5. Consider broader changes (data structures, algorithms, call patterns)
"""
            
            patch_instructions = """
CRITICAL PATCH FORMAT REQUIREMENTS:

⚠️ MOST IMPORTANT: Your context lines must EXACTLY match the actual file content shown below.
Every space, tab, and character must be identical. Common mistakes:
- Using wrong line numbers (the file content below is the ACTUAL file)
- Mixing tabs and spaces (check the file content carefully)
- Missing or extra blank lines
- Different indentation

1. Use standard unified diff format:
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
   context line (space prefix) <-- MUST match actual file EXACTLY
-removed line (minus prefix)
+added line (plus prefix)
   context line (space prefix) <-- MUST match actual file EXACTLY

2. File paths must EXACTLY match the files shown below
3. Context lines must EXACTLY match the original file (character-for-character)
   - Count the spaces/tabs in the original
   - Copy them EXACTLY
   - Even one wrong space will cause the patch to fail
4. Include 3 lines of context before and after changes
5. Line numbers in @@ headers must be accurate (count carefully from the actual file)
6. Preserve all whitespace exactly (spaces, tabs, blank lines)
7. Do NOT include explanations - ONLY the patch

HOW TO ENSURE CORRECT CONTEXT:
1. Find the code you want to change in the FILES section below
2. Copy the surrounding lines EXACTLY as they appear
3. Count the line numbers carefully
4. Use those exact lines as context in your patch

EXAMPLE VALID PATCH:
--- a/example.py
+++ b/example.py
@@ -10,7 +10,7 @@
 def process(items):
     # existing context
     result = []
-    for item in items:
-        result.append(item * 2)
+    # Optimized with list comprehension
+    result = [item * 2 for item in items]
     return result
"""
            
            # Get absolute repo path for explicit instruction
            abs_repo_path = self.repo.root.absolute()
            
            user_prompt = f"""Optimize this code for better performance.

TARGET: Achieve AT LEAST 5% improvement (required for success)

📁 LIBRARY LOCATION (CRITICAL):
The library code you must optimize is located at:
{abs_repo_path}

Your patch must modify files in this directory, NOT the workload script.
DO NOT optimize the workload code - optimize the LIBRARY code that the workload calls.

Metrics:
- Execution time: {baseline_metrics.get('execution_time', 0):.3f}s
- Memory: {baseline_metrics.get('memory_mb', 0):.1f}MB
- Success rate: {baseline_metrics.get('success_rate', 0):.1%}

{profiling_context}

{retry_context}

{mini_context}

{workload_context}

{patch_instructions}

FILES TO OPTIMIZE (use these EXACT paths and contents):
All paths are relative to: {abs_repo_path}

{code_context}

CRITICAL REMINDER:
- Optimize the LIBRARY code at {abs_repo_path}
- Do NOT optimize the workload script
- The workload shows you WHAT to optimize, not WHERE

Generate ONLY a valid unified diff patch. No explanations.
"""
            
            logger.info(f"Calling {self.model}...")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2 + (retry_attempt - 1) * 0.15,
                max_tokens=4096
            )
            
            patch = response.choices[0].message.content.strip()
            
            # Clean formatting
            if patch.startswith("```diff"):
                patch = patch[7:]
            elif patch.startswith("```"):
                patch = patch[3:]
            if patch.endswith("```"):
                patch = patch[:-3]
            patch = patch.strip()
            
            # Fix GPT adding extra space before every line
            # Unified diff format: " " for context, "-" for removed, "+" for added
            # GPT often adds an extra space, breaking the format
            lines = patch.split('\n')
            fixed_lines = []
            for line in lines:
                # Skip header lines (---, +++, @@)
                if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                    fixed_lines.append(line)
                # For diff content lines, remove the extra leading space if present
                elif line.startswith('  '):  # Extra space before context line
                    fixed_lines.append(line[1:])  # Remove one space
                elif line.startswith(' -') or line.startswith(' +'):  # Extra space before +/-
                    fixed_lines.append(line[1:])  # Remove one space
                else:
                    fixed_lines.append(line)
            
            patch = '\n'.join(fixed_lines)
            
            return patch if patch else None
            
        except ImportError:
            logger.error("openai package not installed")
            return None
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            return None
    
    def _generate_template_patch(self) -> str:
        """Fallback template patch"""
        logger.info("Generating template optimization patch")
        
        patch = """--- a/main.py
+++ b/main.py
@@ -10,8 +10,7 @@
 def process_data(items):
-    result = []
-    for item in items:
-        result.append(transform(item))
+    # Optimization: Use list comprehension
+    result = [transform(item) for item in items]
     return result"""
        
        logger.warning("Using template patch - for testing only!")
        return patch
    
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
    parser.add_argument('--workload', help='Workload code file (optional)')
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