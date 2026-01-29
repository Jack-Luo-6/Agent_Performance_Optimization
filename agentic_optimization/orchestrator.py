"""
Performance Optimization Architecture - Orchestrator (Refactored & Fixed)
Co-evolution of code optimization and workload stress testing

FIXES:
1. Optimized code must beat baseline on BOTH baseline workload AND current workload
2. Correctness only needs to match baseline (not necessarily perfect)
3. Ultimate success measured by baseline workload improvement
4. First run excluded from baseline (warmup/cache)
5. NEW: When Mini-SWE-Agent succeeds, skip patch application (already applied)
6. NEW: WorkloadGenerator now receives and analyzes baseline workload
"""

import json
import logging
import sqlite3
import time
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tools.profiler import Profiler

sys.path.insert(0, str(Path(__file__).parent))

from opencode.repo import OpenCodeRepo
from mcp.server import MCPServer
from agents.optimizer_agent import CodeOptimizer
from agents.workload_agent import WorkloadGenerator
from tools.detailed_logger import DetailedLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('Orchestrator')


@dataclass
class RunMetrics:
    """Metrics from a single run"""
    execution_time: float
    p50_time: float
    p99_time: float
    memory_mb: float
    cpu_percent: float
    success_rate: float
    correctness_pass: bool = True
    correctness_details: str = ""
    
    def is_better_than(self, other: 'RunMetrics', threshold: float = 0.05) -> bool:
        """
        Check if significantly better (faster)
        
        Args:
            other: Baseline metrics to compare against
            threshold: Minimum improvement ratio (default 5%)
        
        Returns:
            True if this is at least threshold% faster than other
        """
        # Don't consider improvements if success rate is too low
        if self.success_rate < 0.95:
            return False
        
        improvement = (other.execution_time - self.execution_time) / other.execution_time
        return improvement > threshold
    
    def is_worse_than(self, other: 'RunMetrics', threshold: float = 0.05) -> bool:
        """Check if significantly worse (slower) - used for workload validation"""
        degradation = (self.execution_time - other.execution_time) / other.execution_time
        return degradation > threshold
    
    def improvement_percent(self, baseline: 'RunMetrics') -> float:
        """Calculate improvement percentage over baseline"""
        return ((baseline.execution_time - self.execution_time) / baseline.execution_time) * 100


@dataclass
class CycleResult:
    """Results from one full cycle"""
    iteration: int
    baseline_workload_code: str
    current_workload_code: str
    optimized_code_patch: Optional[str]
    baseline_metrics: RunMetrics
    optimized_on_baseline_metrics: Optional[RunMetrics]  # Performance on baseline workload
    optimized_on_current_metrics: Optional[RunMetrics]   # Performance on current workload
    workload_accepted: bool
    code_accepted: bool
    baseline_improvement_percent: float = 0.0  # THE key metric
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class Database:
    """SQLite database for run history"""
    
    def __init__(self, db_path: str = "runs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
        logger.info(f"📊 Database initialized: {db_path}")
    
    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycles (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                baseline_workload_code TEXT,
                current_workload_code TEXT,
                optimized_code_patch TEXT,
                baseline_metrics TEXT,
                optimized_on_baseline_metrics TEXT,
                optimized_on_current_metrics TEXT,
                workload_accepted BOOLEAN,
                code_accepted BOOLEAN,
                baseline_improvement_percent REAL,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                phase TEXT,
                message TEXT,
                details TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()
    
    def save_cycle(self, cycle: CycleResult) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO cycles 
            (iteration, baseline_workload_code, current_workload_code,
             optimized_code_patch, baseline_metrics, optimized_on_baseline_metrics,
             optimized_on_current_metrics, workload_accepted, code_accepted,
             baseline_improvement_percent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle.iteration,
            cycle.baseline_workload_code,
            cycle.current_workload_code,
            cycle.optimized_code_patch,
            json.dumps(asdict(cycle.baseline_metrics)),
            json.dumps(asdict(cycle.optimized_on_baseline_metrics)) if cycle.optimized_on_baseline_metrics else None,
            json.dumps(asdict(cycle.optimized_on_current_metrics)) if cycle.optimized_on_current_metrics else None,
            cycle.workload_accepted,
            cycle.code_accepted,
            cycle.baseline_improvement_percent,
            cycle.timestamp
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def log(self, iteration: int, phase: str, message: str, details: str = ""):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO logs (iteration, phase, message, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (iteration, phase, message, details, datetime.now().isoformat()))
        self.conn.commit()
    
    def close(self):
        self.conn.close()


class Orchestrator:
    """Main orchestration loop - co-evolving code and workloads"""
    
    def __init__(self, 
                 target_repo: str,
                 baseline_workload: str,
                 correctness_test_dir: str,
                 max_iterations: int = 3,
                 max_code_retries: int = 3,
                 max_workload_retries: int = 3,
                 model: str = "gpt-4o"):
        """
        Initialize orchestrator
        
        Args:
            target_repo: Path to code repository
            baseline_workload: Path to baseline stress test workload
            correctness_test_dir: Path to correctness test script directory
            max_iterations: Number of co-evolution cycles
            max_code_retries: Max retries for code optimizer
            max_workload_retries: Max retries for workload generator
            model: LLM model name
        """
        # Install dependencies first
        self._install_dependencies()
        
        self.target_repo_path = Path(target_repo)
        self.baseline_workload_path = Path(baseline_workload)
        self.correctness_test_dir = Path(correctness_test_dir)
        self.max_iterations = max_iterations
        self.max_code_retries = max_code_retries
        self.max_workload_retries = max_workload_retries
        self.model = model
        
        # Infrastructure
        self.mcp = MCPServer()
        self.db = Database()
        self.profiler = Profiler(self.target_repo_path)
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize detailed text logger
        repo_name = Path(target_repo).name or "unknown_repo"
        cmd_args = {
            'repo': target_repo,
            'baseline_workload': baseline_workload,
            'correctness_tests': correctness_test_dir,
            'iterations': max_iterations,
            'code_retries': max_code_retries,
            'workload_retries': max_workload_retries,
            'model': model
        }
        self.detailed_log = DetailedLogger(repo_name, cmd_args)
        
        # State tracking
        self.baseline_workload_code = self.baseline_workload_path.read_text()
        self.baseline_metrics = None  # Set during initialization - THE reference point
        self.baseline_correctness_pass = None  # Baseline correctness status
        self.current_workload_code = self.baseline_workload_code
        self.current_best_on_baseline = None  # Best performance on BASELINE workload
        self.current_best_on_current = None   # Best performance on CURRENT workload
        
        logger.info("="*80)
        logger.info("🚀 ORCHESTRATOR INITIALIZED")
        logger.info("="*80)
        logger.info(f"Target repo: {self.target_repo_path}")
        logger.info(f"Baseline workload: {self.baseline_workload_path}")
        logger.info(f"Correctness tests: {self.correctness_test_dir}")
        logger.info(f"Model: {model}")
        logger.info(f"Max code retries: {max_code_retries}")
        logger.info(f"Max workload retries: {max_workload_retries}")
        logger.info("="*80)
    
    def _install_dependencies(self):
        """Reinstall dependencies from requirements.txt (uninstall first, then install)"""
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            logger.warning("⚠️  No requirements.txt found, skipping dependency installation")
            return
        
        logger.info("📦 Reinstalling dependencies from requirements.txt...")
        
        try:
            # Force reinstall all packages listed in requirements.txt
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '-r', 'requirements.txt'],
                capture_output=True,
                text=True,
                timeout=600  # increased timeout for reinstall
            )
            
            if result.returncode == 0:
                logger.info("✓ Dependencies reinstalled successfully")
                # Log key installed packages
                installed = [line.strip() for line in result.stdout.split('\n') 
                            if 'Successfully installed' in line or 'Requirement already satisfied' in line]
                if installed:
                    logger.info(f"   {installed[0][:100]}...")
            else:
                logger.error(f"❌ Failed to reinstall dependencies: {result.stderr[:500]}")
                logger.warning("⚠️  Continuing anyway, but some features may not work")
        
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency reinstallation timed out after 10 minutes")
            logger.warning("⚠️  Continuing anyway, but some features may not work")
        except Exception as e:
            logger.error(f"❌ Error reinstalling dependencies: {e}")
            logger.warning("⚠️  Continuing anyway, but some features may not work")

    
    def run(self):
        """Main execution loop"""
        try:
            # Phase 0: Establish baseline
            if not self._establish_baseline():
                logger.error("❌ Failed to establish baseline")
                return
            
            # Phase 1-N: Co-evolution cycles
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 CYCLE {iteration}/{self.max_iterations}")
                logger.info(f"{'='*80}\n")
                
                success = self._run_cycle(iteration)
                
                if not success:
                    logger.warning(f"⚠️  Cycle {iteration} failed - terminating")
                    break
                
                time.sleep(1)
            
            self._print_summary()
            
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Orchestrator stopped by user")
            self._print_summary()
        except Exception as e:
            logger.error(f"💥 Fatal error: {e}", exc_info=True)
        finally:
            self.profiler.cleanup()
            self.db.close()
    
    def _establish_baseline(self) -> bool:
        """Phase 0: Run original code on baseline workload + correctness"""
        logger.info("\n" + "="*80)
        logger.info("📏 PHASE 0: ESTABLISHING BASELINE")
        logger.info("="*80 + "\n")
        
        # Test baseline workload (EXCLUDING FIRST RUN FOR WARMUP)
        logger.info("🧪 Testing original code on baseline workload (with warmup)...")
        baseline_perf = self._run_workload_on_repo(
            self.baseline_workload_code,
            self.target_repo_path,
            "baseline_workload",
            exclude_first_run=True  # NEW: Exclude warmup run
        )
        
        if not baseline_perf:
            logger.error("❌ Failed to run baseline workload")
            return False
        
        # Test correctness
        logger.info("✅ Testing original code correctness...")
        correctness_pass, correctness_details = self._run_correctness_tests(
            self.target_repo_path
        )
        
        baseline_perf.correctness_pass = correctness_pass
        baseline_perf.correctness_details = correctness_details
        
        # Store baseline state
        self.baseline_metrics = baseline_perf
        self.baseline_correctness_pass = correctness_pass
        self.current_best_on_baseline = baseline_perf
        self.current_best_on_current = baseline_perf  # Initially same
        
        logger.info("\n" + "="*80)
        logger.info("📊 BASELINE ESTABLISHED")
        logger.info("="*80)
        logger.info(f"⏱️  Execution time: {baseline_perf.execution_time:.3f}s (warmup excluded)")
        logger.info(f"📈 P50: {baseline_perf.p50_time:.3f}s | P99: {baseline_perf.p99_time:.3f}s")
        logger.info(f"💾 Memory: {baseline_perf.memory_mb:.1f}MB")
        logger.info(f"🎯 Success rate: {baseline_perf.success_rate:.1%}")
        logger.info(f"✅ Correctness: {'PASS' if correctness_pass else 'FAIL'}")
        if correctness_details:
            logger.info(f"   Details: {correctness_details}")
        logger.info("="*80 + "\n")
        
        # Log to detailed logger
        with OpenCodeRepo(str(self.target_repo_path), mode="direct") as repo:
            original_code_files = repo.read_all_code(max_size=50000)
        
        self.detailed_log.log_baseline_info(
            baseline_workload_path=str(self.baseline_workload_path),
            baseline_workload_code=self.baseline_workload_code,
            correctness_test_dir=str(self.correctness_test_dir),
            original_code_files=original_code_files
        )
        
        self.detailed_log.log_baseline_results(
            baseline_metrics=baseline_perf,
            correctness_pass=correctness_pass,
            correctness_details=correctness_details
        )
        
        self.db.log(0, "baseline", "Baseline established", json.dumps(asdict(baseline_perf)))
        
        # Note: We don't require baseline to pass correctness - we just track it
        return True
    
    def _is_mini_swe_patch(self, patch: str) -> bool:
        """
        Check if patch is a Mini-SWE-Agent success marker (not a real patch)
        
        Mini-SWE-Agent applies changes directly, so it returns a marker like:
        # Mini-SWE-Agent made modifications
        # See trajectory: /path/to/trajectory.json
        """
        if not patch:
            return False
        
        # Check for Mini-SWE-Agent markers
        markers = [
            "Mini-SWE-Agent made modifications",
            "Mini-SWE-Agent completed",
            "See trajectory:"
        ]
        
        return any(marker in patch for marker in markers)
    
    def _run_cycle(self, iteration: int) -> bool:
        """
        Single co-evolution cycle
        
        Workflow:
        1. Code Optimization Phase (with retries)
           - Optimize code against current workload
           - Test on BOTH current workload AND baseline workload
           - Must beat baseline on BOTH workloads (key fix!)
           - Correctness must match baseline (not necessarily perfect)
        2. Workload Evolution Phase (with retries)
           - Generate harder workload
           - Validate it's actually harder (slower on optimized code)
        
        Success Criteria:
        - Optimized code must be faster on baseline workload (primary metric)
        - Optimized code must be faster on current workload (secondary metric)
        - Correctness must match baseline correctness
        """
        
        # Log iteration start
        self.detailed_log.log_iteration_start(iteration, self.max_iterations)
        
        # === PHASE 1: CODE OPTIMIZATION ===
        logger.info(f"\n{'─'*80}")
        logger.info(f"🔧 PHASE 1: CODE OPTIMIZATION")
        logger.info(f"{'─'*80}\n")
        
        optimized_code_patch = None
        optimized_on_baseline_metrics = None
        optimized_on_current_metrics = None
        previous_patch_error = None
        
        for code_attempt in range(1, self.max_code_retries + 1):
            # Profile current code before optimization
            logger.info(f"   🔬 Profiling current code on workload...")
            profiling_report = self.profiler.profile_workload(
                self.current_workload_code,
                f"cycle{iteration}_attempt{code_attempt}"
            )
            
            if not profiling_report:
                logger.warning("   ⚠️  Profiling failed, proceeding without profiling data")
            else:
                logger.info(f"   📊 Profiling complete:")
                logger.info(f"      Functions: {len(profiling_report.function_profiles)}")
                logger.info(f"      Hot lines: {len(profiling_report.line_profiles)}")
                logger.info(f"      Peak memory: {profiling_report.memory_profile.peak_mb:.1f} MB")
                logger.info(f"      Coverage: {profiling_report.coverage.overall_coverage_percent:.1f}%")
            
            # Log profiling results to detailed logger
            if profiling_report:
                self.detailed_log.log_profiling_results(profiling_report, iteration, code_attempt)

            logger.info(f"\n🔄 Code optimization attempt {code_attempt}/{self.max_code_retries}")
            
            # Generate optimization
            with OpenCodeRepo(str(self.target_repo_path), mode="direct") as repo:
                optimizer = CodeOptimizer(repo, self.mcp, model=self.model)
                
                patch = optimizer.optimize(
                    asdict(self.current_best_on_current),
                    workload_type=f"cycle_{iteration}",
                    workload_code=self.current_workload_code,
                    retry_attempt=code_attempt,
                    previous_patch_error=previous_patch_error
                )
                
                if not patch:
                    logger.warning(f"   ⚠️  No optimization generated")
                    
                    # Log patch failure
                    self.detailed_log.log_patch_outcome(
                        patch=None,
                        success=False,
                        error="No optimization generated",
                        iteration=iteration,
                        attempt=code_attempt
                    )
                    
                    if code_attempt == self.max_code_retries:
                        self.db.log(iteration, "code_opt", "All code optimization attempts failed", "")
                        return False
                    continue
                
                # Save patch to artifacts
                self._save_patch(iteration, code_attempt, patch)
                
                # **NEW: Check if Mini-SWE-Agent already applied the patch**
                mini_swe_applied = self._is_mini_swe_patch(patch)
                
                if mini_swe_applied:
                    logger.info(f"   🤖 Mini-SWE-Agent already applied changes - skipping patch application")
                    patch_applied = True  # Consider it "applied"
                else:
                    # Log patch outcome
                    self.detailed_log.log_patch_outcome(
                        patch=patch,
                        success=True,
                        error=None,
                        iteration=iteration,
                        attempt=code_attempt
                    )
                    
                    # Apply patch normally
                    logger.info(f"   📝 Applying patch...")
                    patch_applied = repo.apply_patch(patch)
                    
                    if not patch_applied:
                        previous_patch_error = "Patch application failed - likely malformed diff format"
                        logger.error(f"   ❌ Failed to apply patch")
                        
                        # Log patch failure
                        self.detailed_log.log_patch_outcome(
                            patch=patch,
                            success=False,
                            error=previous_patch_error,
                            iteration=iteration,
                            attempt=code_attempt
                        )
                        
                        if code_attempt == self.max_code_retries:
                            return False
                        continue
                
                previous_patch_error = None
                
                # === CRITICAL: Test on BOTH workloads ===
                
                # Test 1: BASELINE workload (PRIMARY METRIC - exclude warmup)
                logger.info(f"   🎯 Testing on BASELINE workload (primary metric)...")
                baseline_test_metrics = self._run_workload_on_repo(
                    self.baseline_workload_code,
                    repo.root,
                    f"optimized_baseline_{iteration}_{code_attempt}",
                    exclude_first_run=True  # Consistent with baseline
                )
                
                if not baseline_test_metrics:
                    logger.error(f"   ❌ Failed to run on baseline workload")
                    if code_attempt == self.max_code_retries:
                        return False
                    continue
                
                # Test 2: CURRENT workload (SECONDARY METRIC)
                logger.info(f"   🧪 Testing on CURRENT workload (secondary metric)...")
                current_test_metrics = self._run_workload_on_repo(
                    self.current_workload_code,
                    repo.root,
                    f"optimized_current_{iteration}_{code_attempt}",
                    exclude_first_run=False  # Current workload doesn't exclude first run
                )
                
                if not current_test_metrics:
                    logger.error(f"   ❌ Failed to run on current workload")
                    if code_attempt == self.max_code_retries:
                        return False
                    continue
                
                # Test 3: Correctness (must match baseline)
                logger.info(f"   ✅ Testing correctness...")
                correctness_pass, correctness_details = self._run_correctness_tests(repo.root)
                baseline_test_metrics.correctness_pass = correctness_pass
                baseline_test_metrics.correctness_details = correctness_details
                
                # Calculate improvements
                baseline_improvement = baseline_test_metrics.improvement_percent(self.baseline_metrics)
                current_improvement = current_test_metrics.improvement_percent(self.current_best_on_current)
                
                logger.info(f"\n   📊 RESULTS:")
                logger.info(f"   {'─'*60}")
                logger.info(f"   🎯 BASELINE workload (PRIMARY):")
                logger.info(f"      Original: {self.baseline_metrics.execution_time:.3f}s")
                logger.info(f"      Optimized: {baseline_test_metrics.execution_time:.3f}s")
                logger.info(f"      Improvement: {baseline_improvement:+.2f}% {'✅' if baseline_improvement > 5.0 else '❌'}")
                logger.info(f"")
                logger.info(f"   🧪 CURRENT workload (SECONDARY):")
                logger.info(f"      Best so far: {self.current_best_on_current.execution_time:.3f}s")
                logger.info(f"      Optimized: {current_test_metrics.execution_time:.3f}s")
                logger.info(f"      Improvement: {current_improvement:+.2f}% {'✅' if current_improvement > 5.0 else '❌'}")
                logger.info(f"")
                logger.info(f"   ✅ Correctness:")
                logger.info(f"      Baseline: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
                logger.info(f"      Optimized: {'PASS' if correctness_pass else 'FAIL'}")
                logger.info(f"      Match: {'✅ YES' if correctness_pass == self.baseline_correctness_pass else '❌ NO'}")
                if correctness_details:
                    logger.info(f"      Details: {correctness_details}")
                logger.info(f"   {'─'*60}\n")
                
                # Log patch test results
                self.detailed_log.log_patch_test_results(
                    baseline_metrics=baseline_test_metrics,
                    current_metrics=current_test_metrics,
                    original_baseline=self.baseline_metrics,
                    original_current=self.current_best_on_current,
                    correctness_pass=correctness_pass,
                    correctness_details=correctness_details,
                    baseline_correctness_pass=self.baseline_correctness_pass,
                    iteration=iteration,
                    attempt=code_attempt
                )
                
                # === VALIDATION: Must beat baseline on BOTH workloads ===
                
                # Check 1: Must beat baseline on BASELINE workload (5% threshold)
                if not baseline_test_metrics.is_better_than(self.baseline_metrics, threshold=0.05):
                    logger.warning(f"   ⚠️  Did not beat baseline on BASELINE workload")
                    logger.warning(f"       Need: >5.0% improvement | Got: {baseline_improvement:+.2f}%")
                    if code_attempt == self.max_code_retries:
                        self.db.log(iteration, "code_opt", "Failed to beat baseline on baseline workload", 
                                   f"Best: {baseline_improvement:.2f}%")
                        return False
                    continue
                
                # Check 2: Must beat baseline on CURRENT workload (5% threshold)
                # (On iteration 1, current == baseline, so this is automatically satisfied)
                if not current_test_metrics.is_better_than(self.current_best_on_current, threshold=0.05):
                    logger.warning(f"   ⚠️  Did not beat baseline on CURRENT workload")
                    logger.warning(f"       Need: >5.0% improvement | Got: {current_improvement:+.2f}%")
                    if code_attempt == self.max_code_retries:
                        self.db.log(iteration, "code_opt", "Failed to beat baseline on current workload",
                                   f"Best: {current_improvement:.2f}%")
                        return False
                    continue
                
                # Check 3: Correctness must match baseline
                if correctness_pass != self.baseline_correctness_pass:
                    logger.warning(f"   ⚠️  Correctness changed from baseline")
                    logger.warning(f"       Baseline: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
                    logger.warning(f"       Optimized: {'PASS' if correctness_pass else 'FAIL'}")
                    if code_attempt == self.max_code_retries:
                        self.db.log(iteration, "code_opt", "Correctness mismatch", 
                                   f"Baseline: {self.baseline_correctness_pass}, Optimized: {correctness_pass}")
                        return False
                    continue
                
                # === SUCCESS! ===
                logger.info(f"   ✅ CODE OPTIMIZATION SUCCESSFUL!")
                logger.info(f"      🎯 Baseline improvement: {baseline_improvement:+.2f}%")
                logger.info(f"      🧪 Current improvement: {current_improvement:+.2f}%")
                logger.info(f"      ✅ Correctness: MATCHES BASELINE")
                
                optimized_code_patch = patch
                optimized_on_baseline_metrics = baseline_test_metrics
                optimized_on_current_metrics = current_test_metrics
                
                # Update current best metrics
                self.current_best_on_baseline = baseline_test_metrics
                self.current_best_on_current = current_test_metrics
                
                self.db.log(iteration, "code_opt", f"Success on attempt {code_attempt}",
                           json.dumps({
                               'baseline_improvement': baseline_improvement,
                               'current_improvement': current_improvement,
                               'correctness_match': True
                           }))
                
                break  # Exit retry loop
        
        if not optimized_code_patch:
            logger.error("❌ Code optimization failed after all retries")
            return False
        
        # === PHASE 2: WORKLOAD EVOLUTION ===
        logger.info(f"\n{'─'*80}")
        logger.info(f"📈 PHASE 2: WORKLOAD EVOLUTION")
        logger.info(f"{'─'*80}\n")
        
        new_workload_code = None
        
        for workload_attempt in range(1, self.max_workload_retries + 1):
            logger.info(f"\n🔄 Workload generation attempt {workload_attempt}/{self.max_workload_retries}")
            
            # Generate harder workload - **NEW: Pass baseline_workload_code**
            workload_gen = WorkloadGenerator(self.mcp, model=self.model)
            
            generated_workload = workload_gen.generate(
                iteration,
                baseline_workload_code=self.baseline_workload_code,  # **NEW: Pass baseline**
                reference_docs=None,
                previous_metrics=asdict(optimized_on_current_metrics)
            )
            
            # Save workload
            self._save_workload(iteration, workload_attempt, generated_workload)
            
            # Validate: Run on optimized code - should be SLOWER
            logger.info(f"   🧪 Validating workload is harder...")
            validation_metrics = self._run_workload_on_repo(
                generated_workload,
                self.target_repo_path,  # Use current optimized code
                f"workload_validation_{iteration}_{workload_attempt}",
                exclude_first_run=False
            )
            
            if not validation_metrics:
                logger.error(f"   ❌ Failed to run validation")
                if workload_attempt == self.max_workload_retries:
                    logger.warning("⚠️  Using current workload (no evolution)")
                    new_workload_code = self.current_workload_code
                    break
                continue
            
            # Check if harder (slower than optimized code on current workload)
            if validation_metrics.is_worse_than(optimized_on_current_metrics, threshold=0.05):
                degradation = ((validation_metrics.execution_time - optimized_on_current_metrics.execution_time) / 
                              optimized_on_current_metrics.execution_time * 100)
                logger.info(f"   ✅ WORKLOAD VALIDATED!")
                logger.info(f"      Optimized on current: {optimized_on_current_metrics.execution_time:.3f}s")
                logger.info(f"      Optimized on new: {validation_metrics.execution_time:.3f}s")
                logger.info(f"      Degradation: {degradation:+.2f}% (harder = good)")
                new_workload_code = generated_workload
                
                # Log new workload
                self.detailed_log.log_new_workload(
                    workload_code=generated_workload,
                    validation_metrics=validation_metrics,
                    optimized_metrics=optimized_on_current_metrics,
                    iteration=iteration,
                    attempt=workload_attempt
                )
                
                self.db.log(iteration, "workload_gen", f"Success on attempt {workload_attempt}",
                           f"Degradation: {degradation:.2f}%")
                break
            else:
                improvement = ((optimized_on_current_metrics.execution_time - validation_metrics.execution_time) / 
                              optimized_on_current_metrics.execution_time * 100)
                logger.warning(f"   ⚠️  Workload not hard enough")
                logger.warning(f"       Need: >5% slower | Got: {improvement:+.2f}% faster")
                if workload_attempt == self.max_workload_retries:
                    logger.warning("⚠️  Using current workload (no evolution)")
                    new_workload_code = self.current_workload_code
                    self.db.log(iteration, "workload_gen", "All attempts failed", 
                               f"Best was {improvement:.2f}% faster")
                    break
        
        # Update state for next cycle
        self.current_workload_code = new_workload_code
        
        # Save cycle result
        cycle = CycleResult(
            iteration=iteration,
            baseline_workload_code=self.baseline_workload_code,
            current_workload_code=new_workload_code,
            optimized_code_patch=optimized_code_patch,
            baseline_metrics=self.baseline_metrics,
            optimized_on_baseline_metrics=optimized_on_baseline_metrics,
            optimized_on_current_metrics=optimized_on_current_metrics,
            workload_accepted=(new_workload_code != self.current_workload_code),
            code_accepted=True,
            baseline_improvement_percent=optimized_on_baseline_metrics.improvement_percent(self.baseline_metrics)
        )
        self.db.save_cycle(cycle)
        
        # Log iteration summary
        self.detailed_log.log_iteration_summary(
            iteration=iteration,
            code_accepted=True,
            workload_accepted=(new_workload_code != self.current_workload_code),
            baseline_improvement=optimized_on_baseline_metrics.improvement_percent(self.baseline_metrics)
        )
        
        logger.info(f"\n✅ CYCLE {iteration} COMPLETE")
        logger.info(f"{'='*80}\n")
        
        return True
    
    def _run_workload_on_repo(self, workload_code: str, repo_path: Path,
                              workload_type: str, exclude_first_run: bool = False) -> Optional[RunMetrics]:
        """
        Run workload on a repository
        
        Args:
            workload_code: Python workload code
            repo_path: Path to repository
            workload_type: Type identifier for logging
            exclude_first_run: If True, exclude first run (warmup/cache)
        """
        try:
            result = self.mcp.run_workload(
                workload_code=workload_code,
                repo_path=repo_path,
                workload_type=workload_type
            )
            
            if result['status'] != 'success':
                logger.error(f"   ❌ Workload failed: {result.get('error', 'unknown')}")
                return None
            
            metrics = result['metrics']
            
            # Handle warmup exclusion if needed
            if exclude_first_run and 'all_runtimes' in metrics and len(metrics['all_runtimes']) > 1:
                # Recalculate excluding first run
                runtimes = metrics['all_runtimes'][1:]
                import statistics
                metrics['execution_time'] = statistics.mean(runtimes)
                if len(runtimes) >= 2:
                    metrics['p50_time'] = statistics.median(runtimes)
                    metrics['p99_time'] = sorted(runtimes)[int(len(runtimes) * 0.99)]
            
            run_metrics = RunMetrics(
                execution_time=metrics['execution_time'],
                p50_time=metrics['p50_time'],
                p99_time=metrics['p99_time'],
                memory_mb=metrics['memory_mb'],
                cpu_percent=metrics['cpu_percent'],
                success_rate=metrics['success_rate']
            )
            
            if exclude_first_run:
                logger.info(f"      (warmup run excluded from timing)")
            
            return run_metrics
        except Exception as e:
            logger.error(f"   ❌ Error running workload: {e}", exc_info=True)
            return None
    
    def _run_correctness_tests(self, repo_path: Path) -> Tuple[bool, str]:
        """Run correctness tests"""
        try:
            # Assume correctness test dir has a run.py or run.sh script
            test_script = None
            for name in ['run.py', 'run.sh', 'test.py']:
                candidate = self.correctness_test_dir / name
                if candidate.exists():
                    test_script = candidate
                    break
            
            if not test_script:
                logger.warning("   ⚠️  No test script found, assuming correctness")
                return True, "No test script"
            
            # Run test script
            cmd = ['python3', str(test_script)] if test_script.suffix == '.py' else ['bash', str(test_script)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            passed = result.returncode == 0
            details = result.stdout[:200] if result.stdout else result.stderr[:200]
            
            return passed, details
            
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            logger.error(f"   ❌ Correctness test error: {e}")
            return False, str(e)
    
    def _save_patch(self, iteration: int, attempt: int, patch: str):
        """Save optimization patch"""
        patch_dir = self.artifacts_dir / "patches"
        patch_dir.mkdir(exist_ok=True)
        
        patch_file = patch_dir / f"cycle{iteration}_attempt{attempt}.diff"
        patch_file.write_text(patch)
        logger.info(f"   💾 Patch saved: {patch_file}")
    
    def _save_workload(self, iteration: int, attempt: int, workload_code: str):
        """Save generated workload"""
        workload_dir = self.artifacts_dir / "workloads"
        workload_dir.mkdir(exist_ok=True)
        
        workload_file = workload_dir / f"cycle{iteration}_attempt{attempt}.py"
        workload_file.write_text(workload_code)
        logger.info(f"   💾 Workload saved: {workload_file}")
    
    def _print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        if self.baseline_metrics and self.current_best_on_baseline:
            total_improvement = self.current_best_on_baseline.improvement_percent(self.baseline_metrics)
            
            logger.info(f"\n🎯 PRIMARY METRIC: BASELINE WORKLOAD PERFORMANCE")
            logger.info(f"   {'─'*60}")
            logger.info(f"   Original (baseline): {self.baseline_metrics.execution_time:.3f}s")
            logger.info(f"   Optimized (final):   {self.current_best_on_baseline.execution_time:.3f}s")
            logger.info(f"   🏆 TOTAL IMPROVEMENT: {total_improvement:+.2f}%")
            logger.info(f"   {'─'*60}")
            
            if self.current_best_on_current != self.current_best_on_baseline:
                logger.info(f"\n🧪 SECONDARY METRIC: CURRENT WORKLOAD PERFORMANCE")
                logger.info(f"   Best on current workload: {self.current_best_on_current.execution_time:.3f}s")
            
            logger.info(f"\n✅ CORRECTNESS STATUS")
            logger.info(f"   Baseline: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
            logger.info(f"   Final:    {'PASS' if self.current_best_on_baseline.correctness_pass else 'FAIL'}")
            logger.info(f"   Match:    {'✅ YES' if self.current_best_on_baseline.correctness_pass == self.baseline_correctness_pass else '❌ NO'}")
        
        logger.info(f"\n📁 ARTIFACTS")
        logger.info(f"   Directory: {self.artifacts_dir}")
        logger.info(f"   Database:  {self.db.db_path}")
        logger.info("="*80)
        
        # Log comprehensive final summary
        self.detailed_log.log_final_summary(
            final_patch=None,
            final_workload=self.current_workload_code,
            baseline_metrics=self.baseline_metrics,
            final_metrics=self.current_best_on_baseline,
            final_on_final_workload=self.current_best_on_current
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimization Orchestrator")
    parser.add_argument('--repo', required=True, help='Target repository path')
    parser.add_argument('--baseline-workload', required=True, help='Baseline stress test workload file')
    parser.add_argument('--correctness-tests', required=True, help='Correctness test directory')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--code-retries', type=int, default=3)
    parser.add_argument('--workload-retries', type=int, default=3)
    parser.add_argument('--model', default='gpt-4o')
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(
        target_repo=args.repo,
        baseline_workload=args.baseline_workload,
        correctness_test_dir=args.correctness_tests,
        max_iterations=args.iterations,
        max_code_retries=args.code_retries,
        max_workload_retries=args.workload_retries,
        model=args.model
    )
    orchestrator.run()