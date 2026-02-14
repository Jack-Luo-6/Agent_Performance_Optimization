"""
Performance Optimization Architecture - Orchestrator (Diverse Workload Version)
Co-evolution of code optimization with DIVERSE workload exploration

NEW ARCHITECTURE:
Phase 1: Optimize on ORIGINAL workload (N iterations)
Phase 2: Generate DIVERSE workloads (agent decides how many) using ConcoLLMic-inspired strategy
Phase 3: For each diverse workload, optimize N iterations (always validating against original baseline)

ConcoLLMic-Inspired Strategy:
- Static CFG analysis + heuristic scoring for path prioritization
- Iterative feedback loop (execute → update scores → repeat)
- Target deep/complex/unexplored paths systematically
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
from dataclasses import dataclass, asdict, field
from tools.profiler import Profiler

sys.path.insert(0, str(Path(__file__).parent))

from opencode.repo import OpenCodeRepo
from mcp.server import MCPServer
from agents.optimizer_agent import CodeOptimizer
from agents.workload_agent import DiverseWorkloadGenerator  # NEW
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
        """Check if significantly better (faster)"""
        if self.success_rate < 0.95:
            return False
        improvement = (other.execution_time - self.execution_time) / other.execution_time
        return improvement > threshold
    
    def is_worse_than(self, other: 'RunMetrics', threshold: float = 0.05) -> bool:
        """Check if significantly worse (slower)"""
        degradation = (self.execution_time - other.execution_time) / other.execution_time
        return degradation > threshold
    
    def improvement_percent(self, baseline: 'RunMetrics') -> float:
        """Calculate improvement percentage over baseline"""
        return ((baseline.execution_time - self.execution_time) / baseline.execution_time) * 100


@dataclass
class WorkloadInfo:
    """Information about a workload"""
    name: str  # e.g., "original", "diverse_1", "diverse_2"
    code: str
    description: str  # Agent's reasoning for this workload
    is_original: bool = False


@dataclass
class OptimizationPhaseResult:
    """Results from optimizing on one workload for N iterations"""
    workload_name: str
    iterations_completed: int
    best_metrics_on_workload: Optional[RunMetrics]  # Best on THIS workload
    best_metrics_on_original: Optional[RunMetrics]  # Best on ORIGINAL workload
    final_patch: Optional[str]
    baseline_improvement_percent: float = 0.0  # vs original baseline
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CycleResult:
    """Results from one full cycle (keeping for compatibility)"""
    iteration: int
    baseline_workload_code: str
    current_workload_code: str
    optimized_code_patch: Optional[str]
    baseline_metrics: RunMetrics
    optimized_on_baseline_metrics: Optional[RunMetrics]
    optimized_on_current_metrics: Optional[RunMetrics]
    workload_accepted: bool
    code_accepted: bool
    baseline_improvement_percent: float = 0.0
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
        
        # Original table (keep for compatibility)
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
        
        # New table for optimization phases
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_phases (
                phase_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workload_name TEXT,
                iterations_completed INTEGER,
                best_metrics_on_workload TEXT,
                best_metrics_on_original TEXT,
                final_patch TEXT,
                baseline_improvement_percent REAL,
                timestamp TEXT
            )
        """)
        
        # New table for diverse workloads
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diverse_workloads (
                workload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                code TEXT,
                description TEXT,
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
        """Keep for compatibility"""
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
    
    def save_optimization_phase(self, phase: OptimizationPhaseResult) -> int:
        """Save optimization phase result"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO optimization_phases
            (workload_name, iterations_completed, best_metrics_on_workload,
             best_metrics_on_original, final_patch, baseline_improvement_percent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            phase.workload_name,
            phase.iterations_completed,
            json.dumps(asdict(phase.best_metrics_on_workload)) if phase.best_metrics_on_workload else None,
            json.dumps(asdict(phase.best_metrics_on_original)) if phase.best_metrics_on_original else None,
            phase.final_patch,
            phase.baseline_improvement_percent,
            phase.timestamp
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def save_diverse_workload(self, workload: WorkloadInfo) -> int:
        """Save diverse workload"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO diverse_workloads (name, code, description, timestamp)
            VALUES (?, ?, ?, ?)
        """, (workload.name, workload.code, workload.description, datetime.now().isoformat()))
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
    """Main orchestration loop - diverse workload exploration"""
    
    def __init__(self, 
                 target_repo: str,
                 baseline_workload: str,
                 correctness_test_dir: str,
                 iterations_per_workload: int = 3,
                 max_code_retries: int = 3,
                 model: str = "gpt-4o"):
        """
        Initialize orchestrator
        
        Args:
            target_repo: Path to code repository
            baseline_workload: Path to baseline stress test workload
            correctness_test_dir: Path to correctness test script directory
            iterations_per_workload: Number of optimization iterations per workload
            max_code_retries: Max retries for code optimizer per iteration
            model: LLM model name
        """
        # Install dependencies first
        self._install_dependencies()
        
        self.target_repo_path = Path(target_repo)
        self.baseline_workload_path = Path(baseline_workload)
        self.correctness_test_dir = Path(correctness_test_dir)
        self.iterations_per_workload = iterations_per_workload
        self.max_code_retries = max_code_retries
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
            'iterations_per_workload': iterations_per_workload,
            'code_retries': max_code_retries,
            'model': model
        }
        self.detailed_log = DetailedLogger(repo_name, cmd_args)
        
        # State tracking
        self.baseline_workload_code = self.baseline_workload_path.read_text()
        self.baseline_metrics = None  # IMMUTABLE - established in Phase 0
        self.baseline_correctness_pass = None
        
        # NEW: Track all workloads
        self.workloads: List[WorkloadInfo] = []
        
        # NEW: Track optimization results per workload
        self.optimization_results: Dict[str, OptimizationPhaseResult] = {}
        
        # NEW: Track fastest times per workload
        self.fastest_original = None  # Fastest time on original workload (updates)
        self.workload_baselines: Dict[str, RunMetrics] = {}  # Baseline per diverse workload
        self.workload_fastest: Dict[str, RunMetrics] = {}  # Fastest per diverse workload
        
        logger.info("="*80)
        logger.info("🚀 ORCHESTRATOR INITIALIZED (DIVERSE WORKLOAD MODE)")
        logger.info("="*80)
        logger.info(f"Target repo: {self.target_repo_path}")
        logger.info(f"Baseline workload: {self.baseline_workload_path}")
        logger.info(f"Correctness tests: {self.correctness_test_dir}")
        logger.info(f"Model: {model}")
        logger.info(f"Iterations per workload: {iterations_per_workload}")
        logger.info(f"Max code retries per iteration: {max_code_retries}")
        logger.info("="*80)
    
    def _install_dependencies(self):
        """Reinstall dependencies from requirements.txt"""
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            logger.warning("⚠️  No requirements.txt found, skipping dependency installation")
            return
        
        logger.info("📦 Reinstalling dependencies from requirements.txt...")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '-r', 'requirements.txt'],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info("✓ Dependencies reinstalled successfully")
            else:
                logger.error(f"❌ Failed to reinstall dependencies: {result.stderr[:500]}")
                logger.warning("⚠️  Continuing anyway, but some features may not work")
        
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency reinstallation timed out after 10 minutes")
            logger.warning("⚠️  Continuing anyway, but some features may not work")
        except Exception as e:
            logger.error(f"❌ Error reinstalling dependencies: {e}")
            logger.warning("⚠️  Continuing anyway, but some features may not work")
    
    def _create_git_checkpoint(self, name: str) -> Optional[str]:
        """
        Create git checkpoint and return commit hash
        
        Args:
            name: Checkpoint name
        
        Returns:
            Commit hash if successful, None otherwise
        """
        try:
            # Stage all changes
            subprocess.run(
                ['git', 'add', '-A'],
                cwd=self.target_repo_path,
                capture_output=True,
                check=True
            )
            
            # Commit
            subprocess.run(
                ['git', 'commit', '-m', f'checkpoint_{name}'],
                cwd=self.target_repo_path,
                capture_output=True,
                check=True
            )
            
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.target_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hash = result.stdout.strip()
            logger.info(f"   📌 Checkpoint created: {name} ({commit_hash[:8]})")
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"   ⚠️  Failed to create checkpoint: {e}")
            return None
    
    def _revert_to_checkpoint(self, commit_hash: str) -> bool:
        """
        Revert to git checkpoint
        
        Args:
            commit_hash: Commit hash to revert to
        
        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ['git', 'reset', '--hard', commit_hash],
                cwd=self.target_repo_path,
                capture_output=True,
                check=True
            )
            
            logger.info(f"   ↩️  Reverted to checkpoint ({commit_hash[:8]})")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Failed to revert to checkpoint: {e}")
            return False
    
    def run(self):
        """Main execution loop - NEW ARCHITECTURE"""
        try:
            # ===== PHASE 0: Establish Baseline =====
            if not self._establish_baseline():
                logger.error("❌ Failed to establish baseline")
                return
            
            # ===== PHASE 1: Optimize on Original Workload =====
            logger.info(f"\n{'='*80}")
            logger.info(f"🎯 PHASE 1: OPTIMIZE ON ORIGINAL WORKLOAD")
            logger.info(f"{'='*80}\n")
            
            original_workload = WorkloadInfo(
                name="original",
                code=self.baseline_workload_code,
                description="Original baseline workload",
                is_original=True
            )
            self.workloads.append(original_workload)
            
            phase1_result = self._optimize_on_workload(
                workload=original_workload,
                phase_name="Phase 1"
            )
            
            if not phase1_result:
                logger.error("❌ Phase 1 failed - cannot proceed")
                return
            
            self.optimization_results["original"] = phase1_result
            
            # ===== PHASE 2: Generate Diverse Workloads =====
            logger.info(f"\n{'='*80}")
            logger.info(f"🌈 PHASE 2: GENERATE DIVERSE WORKLOADS")
            logger.info(f"{'='*80}\n")
            
            diverse_workloads = self._generate_diverse_workloads()
            
            if not diverse_workloads:
                logger.warning("⚠️  No diverse workloads generated - ending optimization")
                self._print_summary()
                return
            
            logger.info(f"✓ Generated {len(diverse_workloads)} diverse workload(s)")
            self.workloads.extend(diverse_workloads)
            
            # ===== PHASE 3: Optimize on Each Diverse Workload =====
            for idx, workload in enumerate(diverse_workloads, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"🔧 PHASE 3.{idx}: OPTIMIZE ON {workload.name.upper()}")
                logger.info(f"{'='*80}")
                logger.info(f"Description: {workload.description}")
                logger.info(f"{'='*80}\n")
                
                phase_result = self._optimize_on_workload(
                    workload=workload,
                    phase_name=f"Phase 3.{idx}"
                )
                
                if phase_result:
                    self.optimization_results[workload.name] = phase_result
                else:
                    logger.warning(f"⚠️  Phase 3.{idx} ({workload.name}) failed - continuing with next workload")
            
            # ===== FINAL EVALUATION: Test on Original Workload =====
            logger.info(f"\n{'='*80}")
            logger.info(f"🏁 FINAL EVALUATION: ORIGINAL WORKLOAD PERFORMANCE")
            logger.info(f"{'='*80}\n")
            
            final_metrics = self._run_workload_on_repo(
                self.baseline_workload_code,
                self.target_repo_path,
                "final_evaluation",
                exclude_first_run=True
            )
            
            if final_metrics:
                final_improvement = final_metrics.improvement_percent(self.baseline_metrics)
                logger.info(f"\n📊 FINAL RESULTS:")
                logger.info(f"   {'─'*60}")
                logger.info(f"   Original baseline: {self.baseline_metrics.execution_time:.3f}s")
                logger.info(f"   Final performance: {final_metrics.execution_time:.3f}s")
                logger.info(f"   🏆 TOTAL IMPROVEMENT: {final_improvement:+.2f}%")
                logger.info(f"   {'─'*60}\n")
                
                # Update fastest if this is the best
                if final_metrics.is_better_than(self.fastest_original):
                    logger.info(f"   ✨ New record! Updating fastest_original")
                    self.fastest_original = final_metrics
            else:
                logger.warning("⚠️  Final evaluation failed")
            
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
            exclude_first_run=True
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
        
        # Store baseline state (IMMUTABLE)
        self.baseline_metrics = baseline_perf
        self.baseline_correctness_pass = correctness_pass
        
        # Initialize fastest metrics (MUTABLE - updates with improvements)
        self.fastest_original = baseline_perf
        
        logger.info("\n" + "="*80)
        logger.info("📊 BASELINE ESTABLISHED (IMMUTABLE REFERENCE)")
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
        
        return True
    
    def _optimize_on_workload(self, workload: WorkloadInfo, phase_name: str) -> Optional[OptimizationPhaseResult]:
        """
        Optimize code for N iterations on a single workload
        
        NEW LOGIC:
        - For original workload: baseline already exists from Phase 0
        - For diverse workload: run once to establish baseline first
        - Track fastest time per workload (updates on improvement)
        - Always validate against original baseline
        - Track if beats fastest_original (informational)
        
        Each iteration:
        1. Create git checkpoint
        2. Profile current code on this workload
        3. Optimize code
        4. Test on THIS workload (must beat fastest for this workload)
        5. Test on ORIGINAL workload (must beat baseline_original)
        6. Check if beats fastest_original (informational, update if so)
        7. Validate correctness
        8. If fail: revert to checkpoint
        9. If success: update fastest metrics, commit
        
        Returns:
            OptimizationPhaseResult if successful, None otherwise
        """
        logger.info(f"Starting {phase_name}: {self.iterations_per_workload} iterations on '{workload.name}'")
        
        # ===== ESTABLISH BASELINE FOR THIS WORKLOAD =====
        if not workload.is_original:
            logger.info(f"\n🎯 Establishing baseline for '{workload.name}'...")
            
            diverse_baseline = self._run_workload_on_repo(
                workload.code,
                self.target_repo_path,
                f"{workload.name}_baseline",
                exclude_first_run=False
            )
            
            if not diverse_baseline:
                logger.error(f"❌ Failed to establish baseline for '{workload.name}'")
                return None
            
            self.workload_baselines[workload.name] = diverse_baseline
            self.workload_fastest[workload.name] = diverse_baseline
            
            logger.info(f"✓ Baseline for '{workload.name}': {diverse_baseline.execution_time:.3f}s")
        else:
            # Original workload uses Phase 0 baseline
            self.workload_baselines[workload.name] = self.baseline_metrics
            self.workload_fastest[workload.name] = self.fastest_original
        
        # ===== OPTIMIZATION ITERATIONS =====
        best_on_workload = None
        best_on_original = None
        final_patch = None
        iterations_completed = 0
        
        for iteration in range(1, self.iterations_per_workload + 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"🔄 {phase_name} - Iteration {iteration}/{self.iterations_per_workload}")
            logger.info(f"{'─'*80}\n")
            
            # Log iteration start (creates new log file)
            self.detailed_log.log_iteration_start(iteration, self.iterations_per_workload, workload.name)
            
            # Create git checkpoint BEFORE optimization
            checkpoint_hash = self._create_git_checkpoint(f"{workload.name}_iter{iteration}_start")
            
            if not checkpoint_hash:
                logger.warning("⚠️  No git checkpoint created - continuing anyway")
            
            # Profile current code on THIS workload
            logger.info(f"🔬 Profiling current code on '{workload.name}'...")
            profiling_report = self.profiler.profile_workload(
                workload.code,
                f"{workload.name}_iter{iteration}"
            )
            
            if profiling_report:
                logger.info(f"📊 Profiling complete:")
                logger.info(f"   Functions: {len(profiling_report.function_profiles)}")
                logger.info(f"   Hot lines: {len(profiling_report.line_profiles)}")
                logger.info(f"   Peak memory: {profiling_report.memory_profile.peak_mb:.1f} MB")
                logger.info(f"   Coverage: {profiling_report.coverage.overall_coverage_percent:.1f}%")
            else:
                logger.warning("⚠️  Profiling failed, proceeding without profiling data")
            
            # Optimization with retries
            optimized_patch = None
            optimized_on_workload_metrics = None
            optimized_on_original_metrics = None
            previous_patch_error = None
            
            for code_attempt in range(1, self.max_code_retries + 1):
                logger.info(f"\n🔄 Code optimization attempt {code_attempt}/{self.max_code_retries}")
                
                # Get current best metrics for this workload
                current_best_on_workload = self.workload_fastest[workload.name]
                
                # Generate optimization
                with OpenCodeRepo(str(self.target_repo_path), mode="direct") as repo:
                    optimizer = CodeOptimizer(repo, self.mcp, model=self.model)
                    
                    patch = optimizer.optimize(
                        asdict(current_best_on_workload),
                        workload_type=workload.name,
                        workload_code=workload.code,
                        retry_attempt=code_attempt,
                        previous_patch_error=previous_patch_error,
                        profiling_report=profiling_report
                    )
                    
                    if not patch:
                        logger.warning(f"   ⚠️  No optimization generated")
                        if code_attempt == self.max_code_retries:
                            logger.error(f"   ❌ All optimization attempts failed for iteration {iteration}")
                            break
                        continue
                    
                    # Save patch
                    self._save_patch(workload.name, iteration, code_attempt, patch)
                    
                    # Check if Mini-SWE-Agent already applied (it modifies files directly)
                    mini_swe_applied = self._is_mini_swe_patch(patch)
                    
                    if mini_swe_applied:
                        logger.info(f"   🤖 Mini-SWE-Agent already applied changes")
                        patch_applied = True
                    else:
                        logger.info(f"   📝 Applying patch...")
                        patch_applied = repo.apply_patch(patch)
                        
                        if not patch_applied:
                            previous_patch_error = "Patch application failed - likely malformed diff format"
                            logger.error(f"   ❌ Failed to apply patch")
                            if code_attempt == self.max_code_retries:
                                break
                            continue
                    
                    previous_patch_error = None
                    
                    # ===== TEST 1: THIS WORKLOAD (must beat fastest for this workload) =====
                    logger.info(f"   🧪 Testing on '{workload.name}' workload...")
                    workload_test_metrics = self._run_workload_on_repo(
                        workload.code,
                        repo.root,
                        f"{workload.name}_optimized_iter{iteration}_attempt{code_attempt}",
                        exclude_first_run=(workload.is_original)
                    )
                    
                    if not workload_test_metrics:
                        logger.error(f"   ❌ Failed to run on '{workload.name}' workload")
                        if checkpoint_hash:
                            self._revert_to_checkpoint(checkpoint_hash)
                        if code_attempt == self.max_code_retries:
                            break
                        continue
                    
                    # ===== TEST 2: ORIGINAL WORKLOAD (must beat baseline_original) =====
                    logger.info(f"   🎯 Testing on ORIGINAL workload (baseline validation)...")
                    original_test_metrics = self._run_workload_on_repo(
                        self.baseline_workload_code,
                        repo.root,
                        f"{workload.name}_on_original_iter{iteration}_attempt{code_attempt}",
                        exclude_first_run=True
                    )
                    
                    if not original_test_metrics:
                        logger.error(f"   ❌ Failed to run on original workload")
                        if checkpoint_hash:
                            self._revert_to_checkpoint(checkpoint_hash)
                        if code_attempt == self.max_code_retries:
                            break
                        continue
                    
                    # ===== TEST 3: CORRECTNESS =====
                    logger.info(f"   ✅ Testing correctness...")
                    correctness_pass, correctness_details = self._run_correctness_tests(repo.root)
                    workload_test_metrics.correctness_pass = correctness_pass
                    workload_test_metrics.correctness_details = correctness_details
                    
                    # ===== CALCULATE IMPROVEMENTS =====
                    workload_improvement = workload_test_metrics.improvement_percent(current_best_on_workload)
                    original_baseline_improvement = original_test_metrics.improvement_percent(self.baseline_metrics)
                    
                    # Check if beats fastest_original (informational)
                    beats_fastest_original = original_test_metrics.is_better_than(self.fastest_original, threshold=0.0)
                    fastest_original_improvement = original_test_metrics.improvement_percent(self.fastest_original) if beats_fastest_original else 0.0
                    
                    # ===== DISPLAY RESULTS =====
                    logger.info(f"\n   📊 RESULTS:")
                    logger.info(f"   {'─'*60}")
                    logger.info(f"   🎯 ORIGINAL workload:")
                    logger.info(f"      Baseline (immutable): {self.baseline_metrics.execution_time:.3f}s")
                    logger.info(f"      Fastest so far: {self.fastest_original.execution_time:.3f}s")
                    logger.info(f"      This optimization: {original_test_metrics.execution_time:.3f}s")
                    logger.info(f"      vs Baseline: {original_baseline_improvement:+.2f}% {'✅ REQUIRED' if original_baseline_improvement > 5.0 else '❌ FAILED'}")
                    if beats_fastest_original:
                        logger.info(f"      vs Fastest: {fastest_original_improvement:+.2f}% ⭐ NEW RECORD!")
                    else:
                        logger.info(f"      vs Fastest: {fastest_original_improvement:+.2f}% (not a record)")
                    logger.info(f"")
                    logger.info(f"   🧪 '{workload.name.upper()}' workload:")
                    logger.info(f"      Baseline: {self.workload_baselines[workload.name].execution_time:.3f}s")
                    logger.info(f"      Fastest so far: {current_best_on_workload.execution_time:.3f}s")
                    logger.info(f"      This optimization: {workload_test_metrics.execution_time:.3f}s")
                    logger.info(f"      Improvement: {workload_improvement:+.2f}% {'✅ REQUIRED' if workload_improvement > 5.0 else '❌ FAILED'}")
                    logger.info(f"")
                    logger.info(f"   ✅ Correctness:")
                    logger.info(f"      Baseline: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
                    logger.info(f"      Optimized: {'PASS' if correctness_pass else 'FAIL'}")
                    logger.info(f"      Match: {'✅ YES' if correctness_pass == self.baseline_correctness_pass else '❌ NO'}")
                    logger.info(f"   {'─'*60}\n")
                    
                    # ===== VALIDATION CHECKS =====
                    
                    # Check 1: Must beat baseline on ORIGINAL workload (REQUIRED)
                    if not original_test_metrics.is_better_than(self.baseline_metrics, threshold=0.05):
                        logger.warning(f"   ⚠️  Did not beat baseline on ORIGINAL workload")
                        logger.warning(f"       Need: >5.0% improvement | Got: {original_baseline_improvement:+.2f}%")
                        if checkpoint_hash:
                            self._revert_to_checkpoint(checkpoint_hash)
                        if code_attempt == self.max_code_retries:
                            logger.error(f"   ❌ Failed to beat baseline on original workload after all retries")
                            break
                        continue
                    
                    # Check 2: Must improve on THIS workload (REQUIRED)
                    if not workload_test_metrics.is_better_than(current_best_on_workload, threshold=0.05):
                        logger.warning(f"   ⚠️  Did not improve on '{workload.name}' workload")
                        logger.warning(f"       Need: >5.0% improvement | Got: {workload_improvement:+.2f}%")
                        if checkpoint_hash:
                            self._revert_to_checkpoint(checkpoint_hash)
                        if code_attempt == self.max_code_retries:
                            logger.error(f"   ❌ Failed to improve on '{workload.name}' after all retries")
                            break
                        continue
                    
                    # Check 3: Correctness must match baseline (REQUIRED)
                    if correctness_pass != self.baseline_correctness_pass:
                        logger.warning(f"   ⚠️  Correctness changed from baseline")
                        logger.warning(f"       Baseline: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
                        logger.warning(f"       Optimized: {'PASS' if correctness_pass else 'FAIL'}")
                        if checkpoint_hash:
                            self._revert_to_checkpoint(checkpoint_hash)
                        if code_attempt == self.max_code_retries:
                            logger.error(f"   ❌ Correctness mismatch after all retries")
                            break
                        continue
                    
                    # ===== SUCCESS! =====
                    logger.info(f"   ✅ OPTIMIZATION SUCCESSFUL!")
                    logger.info(f"      🎯 Original baseline improvement: {original_baseline_improvement:+.2f}%")
                    logger.info(f"      🧪 '{workload.name}' improvement: {workload_improvement:+.2f}%")
                    if beats_fastest_original:
                        logger.info(f"      ⭐ NEW RECORD on original workload: {fastest_original_improvement:+.2f}%")
                    logger.info(f"      ✅ Correctness: MATCHES BASELINE")
                    
                    # Update metrics
                    optimized_patch = patch
                    optimized_on_workload_metrics = workload_test_metrics
                    optimized_on_original_metrics = original_test_metrics
                    
                    # Update fastest times
                    self.workload_fastest[workload.name] = workload_test_metrics
                    if beats_fastest_original:
                        logger.info(f"   ✨ Updating fastest_original: {self.fastest_original.execution_time:.3f}s → {original_test_metrics.execution_time:.3f}s")
                        self.fastest_original = original_test_metrics
                    
                    # Commit successful optimization
                    success_hash = self._create_git_checkpoint(f"{workload.name}_iter{iteration}_success")
                    
                    break  # Exit retry loop
            
            # Check if iteration succeeded
            if not optimized_patch:
                logger.error(f"❌ Iteration {iteration} failed after all retries")
                # Revert to checkpoint if available
                if checkpoint_hash:
                    self._revert_to_checkpoint(checkpoint_hash)
                break  # Exit iteration loop
            
            # Update best metrics
            best_on_workload = optimized_on_workload_metrics
            best_on_original = optimized_on_original_metrics
            final_patch = optimized_patch
            iterations_completed = iteration
            
            logger.info(f"✅ Iteration {iteration} complete")
        
        # Create phase result
        if iterations_completed > 0:
            phase_result = OptimizationPhaseResult(
                workload_name=workload.name,
                iterations_completed=iterations_completed,
                best_metrics_on_workload=best_on_workload,
                best_metrics_on_original=best_on_original,
                final_patch=final_patch,
                baseline_improvement_percent=best_on_original.improvement_percent(self.baseline_metrics) if best_on_original else 0.0
            )
            
            self.db.save_optimization_phase(phase_result)
            
            logger.info(f"\n✅ {phase_name} COMPLETE")
            logger.info(f"   Iterations: {iterations_completed}/{self.iterations_per_workload}")
            logger.info(f"   Baseline improvement: {phase_result.baseline_improvement_percent:+.2f}%")
            logger.info(f"{'='*80}\n")
            
            return phase_result
        else:
            logger.error(f"❌ {phase_name} failed - no successful iterations")
            return None
    
    def _generate_diverse_workloads(self) -> List[WorkloadInfo]:
        """
        Generate diverse workloads using ConcoLLMic-inspired strategy
        
        Agent decides:
        - How many workloads to generate
        - What paths/patterns each workload targets
        
        Returns:
            List of WorkloadInfo objects
        """
        logger.info("🌈 Generating diverse workloads...")
        logger.info("   Agent will decide how many workloads to create")
        
        # Log to detailed logger
        self.detailed_log.log_diverse_workload_generation_start()
        
        # Profile current optimized code
        logger.info("   🔬 Profiling optimized code for diversity analysis...")
        
        # Use the best code from Phase 1
        phase1_result = self.optimization_results.get("original")
        if not phase1_result:
            logger.error("   ❌ No Phase 1 results available")
            return []
        
        profiling_report = self.profiler.profile_workload(
            self.baseline_workload_code,
            "diversity_analysis"
        )
        
        if not profiling_report:
            logger.warning("   ⚠️  Profiling failed - agent will work without profiling data")
        else:
            logger.info(f"   📊 Profiling complete:")
            logger.info(f"      Coverage: {profiling_report.coverage.overall_coverage_percent:.1f}%")
            logger.info(f"      Functions profiled: {len(profiling_report.function_profiles)}")
        
        # Generate diverse workloads
        generator = DiverseWorkloadGenerator(self.mcp, model=self.model)
        
        workload_infos = generator.generate_diverse_workloads(
            baseline_workload_code=self.baseline_workload_code,
            target_repo_path=self.target_repo_path,
            profiling_report=profiling_report,
            baseline_metrics=self.baseline_metrics
        )
        
        if not workload_infos:
            logger.warning("⚠️  No diverse workloads generated")
            return []
        
        # Save workloads
        for workload in workload_infos:
            self._save_workload(workload.name, workload.code)
            self.db.save_diverse_workload(workload)
        
        # Log to detailed logger
        self.detailed_log.log_diverse_workloads_generated(workload_infos)
        
        logger.info(f"✓ Generated {len(workload_infos)} diverse workload(s):")
        for wl in workload_infos:
            logger.info(f"   • {wl.name}: {wl.description}")
        
        return workload_infos
    
    def _is_mini_swe_patch(self, patch: str) -> bool:
        """Check if patch is a Mini-SWE-Agent success marker"""
        if not patch:
            return False
        
        markers = [
            "Mini-SWE-Agent made modifications",
            "Mini-SWE-Agent completed",
            "See trajectory:"
        ]
        
        return any(marker in patch for marker in markers)
    
    def _run_workload_on_repo(self, workload_code: str, repo_path: Path,
                              workload_type: str, exclude_first_run: bool = False) -> Optional[RunMetrics]:
        """Run workload on a repository"""
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
            
            # Handle warmup exclusion
            if exclude_first_run and 'all_runtimes' in metrics and len(metrics['all_runtimes']) > 1:
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
            test_script = None
            for name in ['run.py', 'run.sh', 'test.py']:
                candidate = self.correctness_test_dir / name
                if candidate.exists():
                    test_script = candidate
                    break
            
            if not test_script:
                logger.warning("   ⚠️  No test script found, assuming correctness")
                return True, "No test script"
            
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
    
    def _save_patch(self, workload_name: str, iteration: int, attempt: int, patch: str):
        """Save optimization patch"""
        patch_dir = self.artifacts_dir / "patches" / workload_name
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        patch_file = patch_dir / f"iter{iteration}_attempt{attempt}.diff"
        patch_file.write_text(patch)
        logger.info(f"   💾 Patch saved: {patch_file}")
    
    def _save_workload(self, workload_name: str, workload_code: str):
        """Save generated workload"""
        workload_dir = self.artifacts_dir / "workloads"
        workload_dir.mkdir(exist_ok=True)
        
        workload_file = workload_dir / f"{workload_name}.py"
        workload_file.write_text(workload_code)
        logger.info(f"   💾 Workload saved: {workload_file}")
    
    def _print_summary(self):
        """Print final summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        if not self.baseline_metrics:
            logger.info("No baseline established")
            return
        
        logger.info(f"\n📏 BASELINE (IMMUTABLE REFERENCE):")
        logger.info(f"   Original workload: {self.baseline_metrics.execution_time:.3f}s")
        logger.info(f"   Correctness: {'PASS' if self.baseline_correctness_pass else 'FAIL'}")
        
        logger.info(f"\n🏆 FASTEST TIMES:")
        logger.info(f"   {'─'*60}")
        logger.info(f"   Original workload:")
        logger.info(f"      Baseline: {self.baseline_metrics.execution_time:.3f}s")
        logger.info(f"      Fastest: {self.fastest_original.execution_time:.3f}s")
        logger.info(f"      Improvement: {self.fastest_original.improvement_percent(self.baseline_metrics):+.2f}%")
        
        for workload_name, baseline in self.workload_baselines.items():
            if workload_name == "original":
                continue
            fastest = self.workload_fastest.get(workload_name)
            if fastest:
                logger.info(f"\n   {workload_name}:")
                logger.info(f"      Baseline: {baseline.execution_time:.3f}s")
                logger.info(f"      Fastest: {fastest.execution_time:.3f}s")
                logger.info(f"      Improvement: {fastest.improvement_percent(baseline):+.2f}%")
        
        logger.info(f"\n🎯 OPTIMIZATION RESULTS BY WORKLOAD:")
        logger.info(f"   {'─'*60}")
        
        for workload_name, result in self.optimization_results.items():
            logger.info(f"\n   📦 {workload_name.upper()}:")
            logger.info(f"      Iterations: {result.iterations_completed}/{self.iterations_per_workload}")
            if result.best_metrics_on_original:
                logger.info(f"      Best on original: {result.best_metrics_on_original.execution_time:.3f}s")
                logger.info(f"      Baseline improvement: {result.baseline_improvement_percent:+.2f}%")
            if result.best_metrics_on_workload and not workload_name == "original":
                logger.info(f"      Best on this workload: {result.best_metrics_on_workload.execution_time:.3f}s")
        
        # Overall best
        if self.optimization_results:
            best_overall = max(
                self.optimization_results.values(),
                key=lambda r: r.baseline_improvement_percent if r.best_metrics_on_original else 0.0
            )
            logger.info(f"\n🏆 BEST OVERALL IMPROVEMENT:")
            logger.info(f"   Workload: {best_overall.workload_name}")
            logger.info(f"   Baseline improvement: {best_overall.baseline_improvement_percent:+.2f}%")
            logger.info(f"   Final time: {best_overall.best_metrics_on_original.execution_time:.3f}s")
        
        logger.info(f"\n📁 ARTIFACTS:")
        logger.info(f"   Directory: {self.artifacts_dir}")
        logger.info(f"   Database: {self.db.db_path}")
        logger.info(f"   Workloads tested: {len(self.workloads)}")
        logger.info(f"   Diverse workloads: {len([w for w in self.workloads if not w.is_original])}")
        logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimization Orchestrator (Diverse Workload)")
    parser.add_argument('--repo', required=True, help='Target repository path')
    parser.add_argument('--baseline-workload', required=True, help='Baseline stress test workload file')
    parser.add_argument('--correctness-tests', required=True, help='Correctness test directory')
    parser.add_argument('--iterations-per-workload', type=int, default=3, help='Optimization iterations per workload')
    parser.add_argument('--code-retries', type=int, default=3, help='Max retries per optimization iteration')
    parser.add_argument('--model', default='gpt-4o', help='LLM model')
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(
        target_repo=args.repo,
        baseline_workload=args.baseline_workload,
        correctness_test_dir=args.correctness_tests,
        iterations_per_workload=args.iterations_per_workload,
        max_code_retries=args.code_retries,
        model=args.model
    )
    orchestrator.run()