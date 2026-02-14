"""
Detailed Text Logger for Orchestrator
Creates comprehensive, human-readable logs of the entire optimization process

NEW: Creates separate log files per iteration/phase to prevent overlapping
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict


class DetailedLogger:
    """
    Creates structured, human-readable text logs of the optimization process.
    Logs are saved to logs/<run_id>/ directory with separate files per phase.
    
    Structure:
    logs/
      <repo_name>_run_001/
        00_master.txt               (run info & summary)
        01_baseline.txt             (Phase 0)
        02_phase1_iter1.txt         (Phase 1, iteration 1)
        02_phase1_iter2.txt         (Phase 1, iteration 2)
        02_phase1_iter3.txt         (Phase 1, iteration 3)
        03_diverse_workload_gen.txt (Phase 2)
        04_diverse_1_iter1.txt      (Phase 3.1, iteration 1)
        04_diverse_1_iter2.txt      (Phase 3.1, iteration 2)
        05_diverse_2_iter1.txt      (Phase 3.2, iteration 1)
        ...
        99_final_summary.txt        (Final evaluation)
    """
    
    def __init__(self, repo_name: str, cmd_args: Dict[str, Any]):
        """
        Initialize the detailed logger
        
        Args:
            repo_name: Name of the repository being optimized
            cmd_args: Command line arguments used to run the orchestrator
        """
        self.repo_name = repo_name
        self.cmd_args = cmd_args
        self.start_time = datetime.now()
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Find next available run number
        existing_runs = list(self.logs_dir.glob(f"{repo_name}_run_*"))
        next_num = len(existing_runs) + 1
        
        # Create run-specific directory
        self.run_dir = self.logs_dir / f"{repo_name}_run_{next_num:03d}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Track current log file
        self.current_log_file = None
        
        # Track file numbering
        self.file_counter = 0
        
        # Initialize master log file with header
        self.master_log = self.run_dir / "00_master.txt"
        self.current_log_file = self.master_log
        self._write_header()
    
    def _get_new_log_file(self, name: str) -> Path:
        """
        Get a new log file for a phase/iteration
        
        Args:
            name: Name for the log file (e.g., "baseline", "phase1_iter1")
        
        Returns:
            Path to the new log file
        """
        self.file_counter += 1
        log_file = self.run_dir / f"{self.file_counter:02d}_{name}.txt"
        self.current_log_file = log_file
        
        # Write header for new log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("╔" + "═" * 98 + "╗\n")
            f.write("║" + " " * 98 + "║\n")
            title = f"{name.upper().replace('_', ' ')}"
            padding = (98 - len(title)) // 2
            f.write("║" + " " * padding + title + " " * (98 - padding - len(title)) + "║\n")
            f.write("║" + " " * 98 + "║\n")
            f.write("╚" + "═" * 98 + "╝\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Directory: {self.run_dir}\n\n")
        
        return log_file
    
    def _write(self, content: str, newlines_before: int = 0, newlines_after: int = 1):
        """Write content to current log file with proper formatting"""
        if self.current_log_file is None:
            return
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write('\n' * newlines_before)
            f.write(content)
            f.write('\n' * newlines_after)
    
    def _section(self, title: str, level: int = 1):
        """Write a section header"""
        if level == 1:
            separator = "=" * 100
            self._write(f"\n{separator}", newlines_before=2)
            self._write(f"{title}", newlines_after=0)
            self._write(separator, newlines_before=0, newlines_after=1)
        elif level == 2:
            separator = "─" * 100
            self._write(f"\n{separator}", newlines_before=1)
            self._write(f"{title}", newlines_after=0)
            self._write(separator, newlines_before=0, newlines_after=1)
        elif level == 3:
            self._write(f"\n{'▪' * 50}", newlines_before=1)
            self._write(f"{title}", newlines_after=0)
            self._write(f"{'▪' * 50}", newlines_before=0, newlines_after=1)
    
    def _write_header(self):
        """Write master log file header"""
        self._write("╔" + "═" * 98 + "╗", newlines_before=0)
        self._write("║" + " " * 98 + "║", newlines_after=0)
        title = f"PERFORMANCE OPTIMIZATION LOG: {self.repo_name}"
        padding = (98 - len(title)) // 2
        self._write("║" + " " * padding + title + " " * (98 - padding - len(title)) + "║", newlines_after=0)
        self._write("║" + " " * 98 + "║", newlines_after=0)
        self._write("╚" + "═" * 98 + "╝", newlines_after=2)
        
        self._section("RUN INFORMATION", level=1)
        self._write(f"Start Time:        {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"Repository:        {self.repo_name}")
        self._write(f"Run Directory:     {self.run_dir}")
        
        self._write("\nCommand Line Arguments:", newlines_before=1)
        for key, value in self.cmd_args.items():
            self._write(f"  --{key:<20} {value}")
        
        self._write("\n" + "="*100, newlines_before=2)
        self._write("Individual phase logs are in separate files within this directory.")
        self._write("="*100, newlines_after=2)
    
    def log_baseline_info(self, baseline_workload_path: str, baseline_workload_code: str,
                         correctness_test_dir: str, original_code_files: Dict[str, str]):
        """Log baseline workload and correctness test information"""
        # Create new log file for baseline
        self._get_new_log_file("baseline")
        
        self._section("BASELINE CONFIGURATION", level=1)
        
        # Baseline workload
        self._write("📊 BASELINE STRESS TEST WORKLOAD", newlines_before=1)
        self._write(f"File: {baseline_workload_path}")
        self._write("\nCode:")
        self._write(baseline_workload_code)
        
        # Correctness tests
        self._write("\n✅ CORRECTNESS TEST CONFIGURATION", newlines_before=2)
        self._write(f"Directory: {correctness_test_dir}")
    
    def log_baseline_results(self, baseline_metrics: Any, correctness_pass: bool, 
                            correctness_details: str):
        """Log baseline performance results"""
        self._section("BASELINE PERFORMANCE RESULTS", level=1)
        
        self._write("⏱️  PERFORMANCE METRICS", newlines_before=1)
        self._write(f"  Execution Time:    {baseline_metrics.execution_time:.6f}s  (warmup excluded)")
        self._write(f"  P50 Latency:       {baseline_metrics.p50_time:.6f}s")
        self._write(f"  P99 Latency:       {baseline_metrics.p99_time:.6f}s")
        self._write(f"  Memory Usage:      {baseline_metrics.memory_mb:.2f} MB")
        self._write(f"  CPU Usage:         {baseline_metrics.cpu_percent:.1f}%")
        self._write(f"  Success Rate:      {baseline_metrics.success_rate:.2%}")
        
        self._write("\n✅ CORRECTNESS RESULTS", newlines_before=2)
        self._write(f"  Status:            {'✅ PASS' if correctness_pass else '❌ FAIL'}")
        if correctness_details:
            self._write(f"  Details:           {correctness_details}")
    
    def log_iteration_start(self, iteration: int, max_iterations: int, workload_name: str = "original"):
        """Log the start of an iteration - creates NEW log file"""
        # Create new log file for this iteration
        if workload_name == "original":
            log_name = f"phase1_iter{iteration}"
        else:
            log_name = f"{workload_name}_iter{iteration}"
        
        self._get_new_log_file(log_name)
        
        self._section(f"ITERATION {iteration}/{max_iterations} - {workload_name.upper()}", level=1)
        self._write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", newlines_before=1)
    
    def log_profiling_results(self, profiling_report: Any, iteration: int, attempt: int):
        """Log detailed profiling results"""
        self._section(f"Profiling Results - Attempt {attempt}", level=2)
        
        if not profiling_report:
            self._write("⚠️  No profiling data available", newlines_before=1)
            return
        
        # Use the to_llm_context() method which formats everything
        self._write(profiling_report.to_llm_context())
    
    def log_mini_swe_thoughts(self, mini_swe_context: Optional[str], iteration: int, attempt: int):
        """Log Mini-SWE-Agent's thought process"""
        self._section(f"Mini-SWE-Agent Analysis - Attempt {attempt}", level=2)
        
        if not mini_swe_context:
            self._write("⚠️  No Mini-SWE-Agent context available", newlines_before=1)
            return
        
        self._write("🤖 MINI-SWE-AGENT THOUGHT PROCESS:", newlines_before=1)
        self._write(mini_swe_context)
    
    def log_gpt_thoughts(self, gpt_prompt: Optional[str], gpt_response: Optional[str],
                        iteration: int, attempt: int):
        """Log GPT's optimization thoughts"""
        self._section(f"GPT-4 Analysis - Attempt {attempt}", level=2)
        
        if not gpt_prompt and not gpt_response:
            self._write("⚠️  No GPT-4 analysis available", newlines_before=1)
            return
        
        if gpt_prompt:
            self._write("📤 PROMPT SENT TO GPT-4:", newlines_before=1)
            self._write(gpt_prompt)
        
        if gpt_response:
            self._write("\n📥 GPT-4 RESPONSE:", newlines_before=2)
            self._write(gpt_response)
    
    def log_patch_outcome(self, patch: Optional[str], success: bool, error: Optional[str],
                         iteration: int, attempt: int):
        """Log patch generation outcome"""
        self._section(f"Patch Outcome - Attempt {attempt}", level=2)
        
        if success and patch:
            self._write("✅ PATCH GENERATED SUCCESSFULLY", newlines_before=1)
            self._write("\n📝 PATCH CONTENT:", newlines_before=1)
            self._write(patch)
        else:
            self._write("❌ PATCH GENERATION FAILED", newlines_before=1)
            if error:
                self._write(f"\n⚠️  Error Details:", newlines_before=1)
                self._write(f"  {error}")
    
    def log_patch_test_results(self, baseline_metrics: Any, current_metrics: Any,
                              original_baseline: Any, original_current: Any,
                              correctness_pass: bool, correctness_details: str,
                              baseline_correctness_pass: bool,
                              iteration: int, attempt: int):
        """Log detailed patch testing results"""
        self._section(f"Patch Test Results - Attempt {attempt}", level=2)
        
        # Baseline workload results
        self._write("🎯 BASELINE WORKLOAD PERFORMANCE:", newlines_before=1)
        baseline_improvement = ((original_baseline.execution_time - baseline_metrics.execution_time) 
                               / original_baseline.execution_time * 100)
        
        self._write(f"  Original Time:     {original_baseline.execution_time:.6f}s")
        self._write(f"  Optimized Time:    {baseline_metrics.execution_time:.6f}s")
        self._write(f"  Improvement:       {baseline_improvement:+.2f}%  {'✅' if baseline_improvement > 5.0 else '❌'}")
        self._write(f"  Memory (before):   {original_baseline.memory_mb:.2f} MB")
        self._write(f"  Memory (after):    {baseline_metrics.memory_mb:.2f} MB")
        self._write(f"  Success Rate:      {baseline_metrics.success_rate:.2%}")
        
        # Current workload results
        self._write("\n🧪 CURRENT WORKLOAD PERFORMANCE:", newlines_before=2)
        current_improvement = ((original_current.execution_time - current_metrics.execution_time) 
                              / original_current.execution_time * 100)
        
        self._write(f"  Original Time:     {original_current.execution_time:.6f}s")
        self._write(f"  Optimized Time:    {current_metrics.execution_time:.6f}s")
        self._write(f"  Improvement:       {current_improvement:+.2f}%  {'✅' if current_improvement > 5.0 else '❌'}")
        self._write(f"  Memory (before):   {original_current.memory_mb:.2f} MB")
        self._write(f"  Memory (after):    {current_metrics.memory_mb:.2f} MB")
        self._write(f"  Success Rate:      {current_metrics.success_rate:.2%}")
        
        # Correctness results
        self._write("\n✅ CORRECTNESS VERIFICATION:", newlines_before=2)
        self._write(f"  Baseline Status:   {'✅ PASS' if baseline_correctness_pass else '❌ FAIL'}")
        self._write(f"  Optimized Status:  {'✅ PASS' if correctness_pass else '❌ FAIL'}")
        self._write(f"  Match:             {'✅ YES' if correctness_pass == baseline_correctness_pass else '❌ NO'}")
        if correctness_details:
            self._write(f"  Details:           {correctness_details}")
        
        # Overall verdict
        self._write("\n📊 OVERALL VERDICT:", newlines_before=2)
        meets_baseline = baseline_improvement > 5.0
        meets_current = current_improvement > 5.0
        meets_correctness = correctness_pass == baseline_correctness_pass
        
        if meets_baseline and meets_current and meets_correctness:
            self._write("  ✅ PATCH ACCEPTED - All criteria met!")
        else:
            self._write("  ❌ PATCH REJECTED - Failed criteria:")
            if not meets_baseline:
                self._write(f"     • Baseline improvement insufficient (need >5.0%, got {baseline_improvement:.2f}%)")
            if not meets_current:
                self._write(f"     • Current workload improvement insufficient (need >5.0%, got {current_improvement:.2f}%)")
            if not meets_correctness:
                self._write(f"     • Correctness mismatch (baseline: {baseline_correctness_pass}, optimized: {correctness_pass})")
    
    def log_new_workload(self, workload_code: str, validation_metrics: Any,
                        optimized_metrics: Any, iteration: int, attempt: int):
        """Log newly generated workload and its validation"""
        self._section(f"New Workload Generation - Attempt {attempt}", level=2)
        
        self._write("📈 GENERATED WORKLOAD CODE:", newlines_before=1)
        self._write(workload_code)
        
        # Validation results
        self._write("\n🧪 WORKLOAD VALIDATION:", newlines_before=2)
        degradation = ((validation_metrics.execution_time - optimized_metrics.execution_time) 
                      / optimized_metrics.execution_time * 100)
        
        self._write(f"  Optimized Code on Previous Workload:  {optimized_metrics.execution_time:.6f}s")
        self._write(f"  Optimized Code on New Workload:       {validation_metrics.execution_time:.6f}s")
        self._write(f"  Performance Degradation:               {degradation:+.2f}%  {'✅' if degradation > 5.0 else '❌'}")
        self._write(f"  Memory Impact:                         {validation_metrics.memory_mb - optimized_metrics.memory_mb:+.2f} MB")
        
        if degradation > 5.0:
            self._write("\n  ✅ WORKLOAD ACCEPTED - Successfully harder!", newlines_before=1)
        else:
            self._write("\n  ❌ WORKLOAD REJECTED - Not hard enough (need >5% slower)", newlines_before=1)
    
    def log_diverse_workload_generation_start(self):
        """Log start of diverse workload generation - creates NEW log file"""
        self._get_new_log_file("diverse_workload_gen")
        
        self._section("DIVERSE WORKLOAD GENERATION (PHASE 2)", level=1)
        self._write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", newlines_before=1)
    
    def log_diverse_workloads_generated(self, workloads: list):
        """Log generated diverse workloads"""
        self._section("GENERATED DIVERSE WORKLOADS", level=1)
        
        self._write(f"Total workloads generated: {len(workloads)}", newlines_before=1)
        
        for idx, wl in enumerate(workloads, 1):
            self._write(f"\n{'─'*100}", newlines_before=2)
            self._write(f"WORKLOAD {idx}: {wl.name}")
            self._write(f"{'─'*100}", newlines_after=0)
            self._write(f"\nDescription: {wl.description}")
            self._write(f"\nCode:")
            self._write(wl.code)
    
    def log_iteration_summary(self, iteration: int, code_accepted: bool, 
                             workload_accepted: bool, baseline_improvement: float):
        """Log iteration summary"""
        self._section(f"Iteration Summary", level=2)
        
        self._write("📊 ITERATION RESULTS:", newlines_before=1)
        self._write(f"  Code Optimization:     {'✅ SUCCESS' if code_accepted else '❌ FAILED'}")
        self._write(f"  Workload Evolution:    {'✅ SUCCESS' if workload_accepted else '❌ FAILED'}")
        self._write(f"  Baseline Improvement:  {baseline_improvement:+.2f}%")
        self._write(f"  Completed:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_final_summary(self, final_patch: Optional[str], final_workload: str,
                         baseline_metrics: Any, final_metrics: Any,
                         final_on_final_workload: Any):
        """Log final optimization summary - creates NEW log file"""
        self._get_new_log_file("final_summary")
        
        self._section("FINAL OPTIMIZATION SUMMARY", level=1)
        
        total_improvement = ((baseline_metrics.execution_time - final_metrics.execution_time) 
                            / baseline_metrics.execution_time * 100)
        
        self._write("🏆 OPTIMIZATION ACHIEVEMENTS:", newlines_before=1)
        self._write(f"  Original Baseline Time:    {baseline_metrics.execution_time:.6f}s")
        self._write(f"  Final Optimized Time:      {final_metrics.execution_time:.6f}s")
        self._write(f"  TOTAL IMPROVEMENT:         {total_improvement:+.2f}%")
        self._write(f"  Memory Reduction:          {baseline_metrics.memory_mb - final_metrics.memory_mb:+.2f} MB")
        
        self._write("\n📝 FINAL PATCH:", newlines_before=2)
        if final_patch:
            self._write(final_patch)
        else:
            self._write("  No final patch available")
        
        self._write("\n📈 FINAL WORKLOAD:", newlines_before=2)
        self._write(final_workload)
        
        self._write("\n🎯 FINAL PERFORMANCE:", newlines_before=2)
        self._write(f"  Final Code on Baseline Workload:  {final_metrics.execution_time:.6f}s ({total_improvement:+.2f}%)")
        if final_on_final_workload:
            self._write(f"  Final Code on Final Workload:     {final_on_final_workload.execution_time:.6f}s")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self._write("\n⏱️  EXECUTION SUMMARY:", newlines_before=2)
        self._write(f"  Start Time:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  End Time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"  Duration:      {duration}")
        
        self._write("\n" + "="*100, newlines_before=2)
        self._write("END OF OPTIMIZATION RUN", newlines_after=0)
        self._write("="*100, newlines_after=2)