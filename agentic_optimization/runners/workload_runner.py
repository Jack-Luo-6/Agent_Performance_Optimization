"""
Workload Runner
Executes generated workload code against target repositories
"""

import logging
import subprocess
import time
import tempfile
import statistics
from pathlib import Path
from typing import Dict, Any, List
import psutil

logger = logging.getLogger('WorkloadRunner')


class WorkloadRunner:
    """Executes workload code and collects metrics"""
    
    def run(self, workload_code: str, repo_path: Path, 
            workload_type: str = "original") -> Dict[str, Any]:
        """
        Execute workload against repository
        
        Args:
            workload_code: Python code to execute
            repo_path: Repository path
            workload_type: "original", "human", or "optimized"
        
        Returns:
            Execution results with metrics
        """
        logger.info(f"Running workload (type: {workload_type})")
        
        # Create temporary workload file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            workload_file = Path(f.name)
            f.write(workload_code)
        
        try:
            # Run multiple iterations to collect statistics
            execution_times = []
            memory_samples = []
            cpu_samples = []
            success_count = 0
            iterations = 10  # Number of runs for statistics
            
            for i in range(iterations):
                result = self._run_single(workload_file, repo_path)
                
                if result['success']:
                    success_count += 1
                    execution_times.append(result['execution_time'])
                    memory_samples.append(result['memory_mb'])
                    cpu_samples.append(result['cpu_percent'])
            
            # Calculate statistics
            if not execution_times:
                return {
                    'status': 'failed',
                    'workload_type': workload_type,
                    'metrics': self._error_metrics()
                }
            
            execution_times.sort()
            
            metrics = {
                'execution_time': statistics.mean(execution_times),
                'p50_time': execution_times[len(execution_times) // 2],
                'p99_time': execution_times[int(len(execution_times) * 0.99)] if len(execution_times) > 1 else execution_times[0],
                'min_time': min(execution_times),
                'max_time': max(execution_times),
                'memory_mb': statistics.mean(memory_samples),
                'cpu_percent': statistics.mean(cpu_samples),
                'success_rate': success_count / iterations,
                'iterations': iterations
            }
            
            logger.info(f"✓ Workload completed: {metrics['execution_time']:.3f}s avg")
            
            return {
                'status': 'success',
                'workload_type': workload_type,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Workload execution failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'workload_type': workload_type,
                'error': str(e),
                'metrics': self._error_metrics()
            }
        finally:
            # Cleanup
            if workload_file.exists():
                workload_file.unlink()
    
    def _run_single(self, workload_file: Path, repo_path: Path) -> Dict[str, Any]:
        """Run single workload iteration"""
        
        start_time = time.perf_counter()
        
        try:
            import sys
            import shutil
            
            # Find python3 in PATH (should be venv's python3 if activated)
            python_exe = shutil.which('python3') or sys.executable
            
            process = subprocess.Popen(
                [python_exe, str(workload_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor resources
            try:
                proc = psutil.Process(process.pid)
                max_memory = 0
                cpu_samples = []
                
                while process.poll() is None:
                    try:
                        mem_mb = proc.memory_info().rss / 1024 / 1024
                        cpu_pct = proc.cpu_percent(interval=0.05)
                        max_memory = max(max_memory, mem_mb)
                        cpu_samples.append(cpu_pct)
                    except psutil.NoSuchProcess:
                        break
                
                avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            except:
                max_memory = 0
                avg_cpu = 0

            stdout, stderr = process.communicate(timeout=120)
            elapsed = time.perf_counter() - start_time
            
            return {
                'success': process.returncode == 0,
                'execution_time': elapsed,
                'memory_mb': max_memory,
                'cpu_percent': avg_cpu,
                'stdout': stdout,
                'stderr': stderr
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                'success': False,
                'execution_time': 120.0,
                'memory_mb': 0,
                'cpu_percent': 0,
                'stdout': '',
                'stderr': 'Timeout'
            }
        
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'memory_mb': 0,
                'cpu_percent': 0,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _error_metrics(self) -> Dict[str, float]:
        """Return error metrics"""
        return {
            'execution_time': 999999.0,
            'p50_time': 999999.0,
            'p99_time': 999999.0,
            'min_time': 999999.0,
            'max_time': 999999.0,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'success_rate': 0.0,
            'iterations': 0
        }