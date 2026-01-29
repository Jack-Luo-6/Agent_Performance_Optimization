"""
Profiling and Coverage Module
Collects function-level, line-level, memory profiling, and coverage data
Located in tools/profiler.py

FAULT TOLERANT: Captures partial results and errors instead of failing completely
"""

import cProfile
import pstats
import tracemalloc
import json
import subprocess
import tempfile
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger('Profiler')


@dataclass
class FunctionProfile:
    """Function-level profile data"""
    name: str
    filename: str
    cumulative_time: float
    total_calls: int
    time_per_call: float
    percent_total: float


@dataclass
class LineProfile:
    """Line-level profile data for hot functions"""
    filename: str
    function: str
    line_no: int
    hits: int
    time_seconds: float
    percent_function: float
    source_line: str


@dataclass
class MemoryProfile:
    """Memory profiling data"""
    peak_mb: float
    current_mb: float
    top_allocations: List[Dict]


@dataclass
class CoverageData:
    """Code coverage data"""
    files_executed: List[str]
    functions_executed: List[Dict]
    overall_coverage_percent: float
    uncovered_lines: Dict[str, List[int]]


@dataclass
class ProfilingError:
    """Error information from profiling"""
    phase: str  # 'function', 'line', 'memory', 'coverage'
    error_type: str
    error_message: str
    traceback_str: str
    stdout: str = ""
    stderr: str = ""


@dataclass
class ProfilingReport:
    """Complete profiling report - now includes errors"""
    function_profiles: List[FunctionProfile]
    line_profiles: List[LineProfile]
    memory_profile: MemoryProfile
    coverage: CoverageData
    errors: List[ProfilingError] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if any profiling phase had errors"""
        return len(self.errors) > 0
    
    def to_llm_context(self) -> str:
        """Format profiling data for LLM consumption - CONCISE VERSION"""
        sections = []
        
        # Show errors first if any (CONCISE)
        if self.errors:
            sections.append("=== PROFILING ERRORS ===")
            for err in self.errors:
                sections.append(f"• {err.phase.upper()}: {err.error_message}")
            sections.append("")
        
        # Function-level profiling (TOP 10 ONLY, >5% TIME)
        if self.function_profiles:
            # Filter to only show significant functions (>5% of total time)
            significant_funcs = [fp for fp in self.function_profiles if fp.percent_total >= 5.0][:10]
            
            if significant_funcs:
                sections.append("=== TOP BOTTLENECKS (>5% time) ===")
                for i, fp in enumerate(significant_funcs, 1):
                    sections.append(
                        f"{i}. {fp.name} - {fp.cumulative_time:.2f}s ({fp.percent_total:.1f}%)"
                        f" [{fp.total_calls:,} calls]"
                    )
                    sections.append(f"   📁 {fp.filename}")
                sections.append("")
            else:
                sections.append("=== FUNCTION PROFILING ===")
                sections.append("No single function dominates (all <5% time)")
                sections.append("Top 3 functions:")
                for i, fp in enumerate(self.function_profiles[:3], 1):
                    sections.append(f"{i}. {fp.name} - {fp.cumulative_time:.2f}s ({fp.percent_total:.1f}%)")
                sections.append("")
        else:
            sections.append("=== FUNCTION PROFILING ===")
            sections.append("❌ No data available")
            sections.append("")
        
        # Memory profiling (CONCISE)
        sections.append("=== MEMORY ===")
        if self.memory_profile.peak_mb > 0:
            sections.append(f"Peak: {self.memory_profile.peak_mb:.1f} MB")
            if self.memory_profile.top_allocations:
                sections.append("Top allocations:")
                for i, alloc in enumerate(self.memory_profile.top_allocations[:3], 1):  # Only top 3
                    sections.append(
                        f"  {i}. {alloc['size_mb']:.1f} MB - {Path(alloc['filename']).name}:{alloc['line']}"
                    )
        else:
            sections.append("❌ No data available")
        sections.append("")
        
        # Coverage (CONCISE)
        sections.append("=== COVERAGE ===")
        if self.coverage.overall_coverage_percent > 0:
            sections.append(f"Overall: {self.coverage.overall_coverage_percent:.1f}%")
            sections.append(f"Files executed: {len(self.coverage.files_executed)}")
        else:
            sections.append("❌ No data available")
        sections.append("")
        
        # Optimization guidance (CONCISE)
        sections.append("=== OPTIMIZATION PRIORITY ===")
        
        if self.function_profiles:
            significant_funcs = [fp for fp in self.function_profiles if fp.percent_total >= 5.0]
            if significant_funcs:
                sections.append(f"🎯 FOCUS HERE: {len(significant_funcs)} functions consume {sum(fp.percent_total for fp in significant_funcs):.0f}% of runtime")
                sections.append(f"   Start with: {significant_funcs[0].name} ({significant_funcs[0].percent_total:.0f}%)")
            else:
                sections.append("⚠️  No dominant bottlenecks - look for algorithmic improvements")
        else:
            sections.append("⚠️  Limited profiling data - analyze workload code directly")
        
        return "\n".join(sections)


class Profiler:
    """Main profiler class - FAULT TOLERANT"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="profiler_"))
        logger.info(f"Profiler initialized (temp: {self.temp_dir})")
    
    def profile_workload(self, workload_code: str, workload_name: str = "workload") -> Optional[ProfilingReport]:
        """
        Run complete profiling suite on workload
        
        FAULT TOLERANT: Returns partial results even if some phases fail
        Never returns None - always returns a report (possibly with errors)
        """
        logger.info(f"🔬 Starting comprehensive profiling for {workload_name}")
        
        workload_file = self.temp_dir / f"{workload_name}.py"
        workload_file.write_text(workload_code)
        
        # Initialize empty results
        func_profiles = []
        line_profiles = []
        memory_profile = MemoryProfile(peak_mb=0, current_mb=0, top_allocations=[])
        coverage_data = CoverageData([], [], 0.0, {})
        errors = []
        
        # Function-level profiling (cProfile)
        logger.info("  📊 Running function-level profiling...")
        try:
            func_profiles = self._run_function_profiling(workload_file)
            if func_profiles:
                logger.info(f"     ✓ Found {len(func_profiles)} functions")
            else:
                logger.warning("     ⚠️  No function profiles extracted")
        except Exception as e:
            logger.error(f"     ❌ Function profiling failed: {e}")
            errors.append(ProfilingError(
                phase='function',
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            ))
        
        # Line-level profiling (line_profiler) - SKIP FOR NOW, TOO FLAKY
        # logger.info("  🎯 Running line-level profiling...")
        # try:
        #     line_profiles = self._run_line_profiling(workload_file, func_profiles)
        #     if line_profiles:
        #         logger.info(f"     ✓ Found {len(line_profiles)} hot lines")
        #     else:
        #         logger.warning("     ⚠️  No hot lines found")
        # except Exception as e:
        #     logger.error(f"     ❌ Line profiling failed: {e}")
        #     errors.append(ProfilingError(
        #         phase='line',
        #         error_type=type(e).__name__,
        #         error_message=str(e),
        #         traceback_str=traceback.format_exc()
        #     ))
        
        # Memory profiling (tracemalloc)
        logger.info("  💾 Running memory profiling...")
        try:
            memory_profile = self._run_memory_profiling(workload_file)
            if memory_profile.peak_mb > 0:
                logger.info(f"     ✓ Peak memory: {memory_profile.peak_mb:.1f} MB")
            else:
                logger.warning("     ⚠️  No memory data captured")
        except Exception as e:
            logger.error(f"     ❌ Memory profiling failed: {e}")
            errors.append(ProfilingError(
                phase='memory',
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            ))
        
        # Coverage (coverage.py) - WITH LONGER TIMEOUT
        logger.info("  ✅ Running coverage analysis...")
        try:
            coverage_data = self._run_coverage(workload_file)
            if coverage_data.overall_coverage_percent > 0:
                logger.info(f"     ✓ Coverage: {coverage_data.overall_coverage_percent:.1f}%")
            else:
                logger.warning("     ⚠️  No coverage data")
        except Exception as e:
            logger.error(f"     ❌ Coverage analysis failed: {e}")
            errors.append(ProfilingError(
                phase='coverage',
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            ))
        
        # Always return a report, even if some/all phases failed
        report = ProfilingReport(
            function_profiles=func_profiles,
            line_profiles=line_profiles,
            memory_profile=memory_profile,
            coverage=coverage_data,
            errors=errors
        )
        
        if errors:
            logger.warning(f"✓ Profiling complete with {len(errors)} error(s) - partial results available")
        else:
            logger.info(f"✓ Profiling complete - all phases successful")
        
        return report
    
    def _run_function_profiling(self, workload_file: Path) -> List[FunctionProfile]:
        """Run cProfile and extract top functions"""
        prof_file = self.temp_dir / "profile.prof"
        
        result = subprocess.run(
            ['python', '-m', 'cProfile', '-o', str(prof_file), str(workload_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.repo_path
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"cProfile failed (code {result.returncode}): {result.stderr}")
        
        if not prof_file.exists():
            raise FileNotFoundError(f"Profile output file not created: {prof_file}")
        
        stats = pstats.Stats(str(prof_file))
        stats.sort_stats('cumulative')
        
        profiles = []
        total_time = stats.total_tt
        
        # Get ALL functions, no limit
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, name = func
            
            # Skip built-ins and site-packages
            if '<built-in>' in filename or 'site-packages' in filename:
                continue
            
            try:
                rel_path = Path(filename).relative_to(self.repo_path)
                filename = str(rel_path)
            except:
                pass
            
            profiles.append(FunctionProfile(
                name=name,
                filename=filename,
                cumulative_time=ct,
                total_calls=nc,
                time_per_call=ct/nc if nc > 0 else 0,
                percent_total=(ct/total_time)*100 if total_time > 0 else 0
            ))
        
        # Sort by cumulative time descending
        profiles.sort(key=lambda x: x.cumulative_time, reverse=True)
        
        return profiles
    
    def _run_line_profiling(self, workload_file: Path, func_profiles: List[FunctionProfile]) -> List[LineProfile]:
        """Run line_profiler on hot functions - GRACEFUL FALLBACK"""
        try:
            subprocess.run(['kernprof', '--help'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("line_profiler not installed, skipping")
            return []
        
        if not func_profiles:
            logger.warning("No function profiles available")
            return []
        
        hot_functions = [fp.name for fp in func_profiles[:5] if not fp.name.startswith('<')]
        
        if not hot_functions:
            logger.warning("No suitable hot functions found")
            return []
        
        workload_code = workload_file.read_text()
        
        instrumented_lines = []
        for line in workload_code.split('\n'):
            for hot_func in hot_functions:
                if f'def {hot_func}(' in line:
                    instrumented_lines.append('@profile')
                    break
            instrumented_lines.append(line)
        
        instrumented_file = self.temp_dir / f"instrumented_{workload_file.name}"
        instrumented_file.write_text('\n'.join(instrumented_lines))
        
        lprof_file = instrumented_file.with_suffix('.py.lprof')
        result = subprocess.run(
            ['kernprof', '-l', str(instrumented_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.repo_path
        )
        
        if result.returncode != 0:
            logger.warning(f"kernprof failed: {result.stderr[:200]}")
            return []  # Gracefully skip
        
        if not lprof_file.exists():
            logger.warning("Line profiler output not created")
            return []  # Gracefully skip
        
        result = subprocess.run(
            ['python', '-m', 'line_profiler', str(lprof_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.warning(f"line_profiler display failed")
            return []  # Gracefully skip
        
        line_profiles = []
        current_func = None
        current_file = None
        func_total_time = 0
        
        for line in result.stdout.split('\n'):
            if line.startswith('File:'):
                parts = line.split()
                current_file = parts[1] if len(parts) > 1 else None
                if 'Function:' in line:
                    func_idx = parts.index('Function:')
                    current_func = parts[func_idx + 1] if func_idx + 1 < len(parts) else None
            
            elif 'Total time:' in line:
                try:
                    func_total_time = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            
            elif line.strip() and line[0].isdigit() and current_func and current_file:
                try:
                    parts = line.split()
                    if len(parts) >= 5:
                        line_no = int(parts[0])
                        hits = int(parts[1])
                        time_microsec = float(parts[2])
                        time_sec = time_microsec / 1e6
                        percent = (time_sec / func_total_time * 100) if func_total_time > 0 else 0
                        
                        if percent >= 5.0:
                            source = ' '.join(parts[5:]) if len(parts) > 5 else ''
                            
                            line_profiles.append(LineProfile(
                                filename=current_file,
                                function=current_func,
                                line_no=line_no,
                                hits=hits,
                                time_seconds=time_sec,
                                percent_function=percent,
                                source_line=source
                            ))
                except (ValueError, IndexError):
                    continue
        
        return sorted(line_profiles, key=lambda x: x.time_seconds, reverse=True)
    
    def _run_memory_profiling(self, workload_file: Path) -> MemoryProfile:
        """Run tracemalloc-based memory profiling"""
        workload_code = workload_file.read_text()
        
        instrumented_code = f"""
import tracemalloc
import json

tracemalloc.start()

{workload_code}

current, peak = tracemalloc.get_traced_memory()
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

result = {{
    'current_mb': current / 1e6,
    'peak_mb': peak / 1e6,
    'top_allocations': [
        {{
            'filename': str(stat.traceback[0].filename),
            'line': stat.traceback[0].lineno,
            'size_mb': stat.size / 1e6,
            'count': stat.count
        }}
        for stat in top_stats[:10]
    ]
}}

print('MEMORY_PROFILE_JSON:', json.dumps(result))

tracemalloc.stop()
"""
        
        instrumented_file = self.temp_dir / f"memory_{workload_file.name}"
        instrumented_file.write_text(instrumented_code)
        
        result = subprocess.run(
            ['python', str(instrumented_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.repo_path
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Memory profiling failed: {result.stderr}")
        
        for line in result.stdout.split('\n'):
            if 'MEMORY_PROFILE_JSON:' in line:
                try:
                    json_str = line.split('MEMORY_PROFILE_JSON:')[1].strip()
                    data = json.loads(json_str)
                    
                    filtered_allocs = [
                        alloc for alloc in data['top_allocations']
                        if 'site-packages' not in alloc['filename'] and '<' not in alloc['filename']
                    ]
                    
                    return MemoryProfile(
                        peak_mb=data['peak_mb'],
                        current_mb=data['current_mb'],
                        top_allocations=filtered_allocs
                    )
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Failed to parse memory JSON: {e}")
        
        raise RuntimeError(f"No MEMORY_PROFILE_JSON marker found")
    
    def _run_coverage(self, workload_file: Path) -> CoverageData:
        """Run coverage.py - WITH EXTENDED TIMEOUT"""
        try:
            subprocess.run(['coverage', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("coverage.py not installed, skipping")
            return CoverageData([], [], 0.0, {})
        
        cov_data_file = self.temp_dir / ".coverage"
        
        result = subprocess.run(
            ['coverage', 'run', '--data-file', str(cov_data_file), str(workload_file)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.repo_path
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Coverage run failed: {result.stderr}")
        
        # INCREASED TIMEOUT: 30s -> 120s for large codebases
        result = subprocess.run(
            ['coverage', 'report', '--data-file', str(cov_data_file)],
            capture_output=True,
            text=True,
            timeout=120,  # <--- INCREASED
            cwd=self.repo_path
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Coverage report failed: {result.stderr}")
        
        files_executed = []
        overall_coverage = 0.0
        
        for line in result.stdout.split('\n'):
            if line.strip() and not line.startswith('Name') and not line.startswith('---'):
                parts = line.split()
                if len(parts) >= 4:
                    filename = parts[0]
                    if 'site-packages' not in filename and not filename.startswith('/'):
                        files_executed.append(filename)
            
            if line.startswith('TOTAL'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        overall_coverage = float(parts[3].rstrip('%'))
                    except:
                        pass
        
        # ALSO INCREASED JSON TIMEOUT
        result = subprocess.run(
            ['coverage', 'json', '--data-file', str(cov_data_file), '-o', str(self.temp_dir / 'coverage.json')],
            capture_output=True,
            text=True,
            timeout=120,  # <--- INCREASED
            cwd=self.repo_path
        )
        
        functions_executed = []
        uncovered_lines = {}
        
        if result.returncode == 0:
            try:
                with open(self.temp_dir / 'coverage.json') as f:
                    cov_json = json.load(f)
                
                for filename, file_data in cov_json.get('files', {}).items():
                    if 'site-packages' in filename:
                        continue
                    
                    missing_lines = file_data.get('missing_lines', [])
                    
                    if missing_lines:
                        uncovered_lines[filename] = missing_lines
                    
                    summary = file_data.get('summary', {})
                    functions_executed.append({
                        'name': Path(filename).stem,
                        'filename': filename,
                        'lines_covered': summary.get('covered_lines', 0),
                        'lines_total': summary.get('num_statements', 0)
                    })
            except Exception as e:
                raise RuntimeError(f"Failed to parse coverage JSON: {e}")
        
        return CoverageData(
            files_executed=files_executed,
            functions_executed=functions_executed,
            overall_coverage_percent=overall_coverage,
            uncovered_lines=uncovered_lines
        )
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")