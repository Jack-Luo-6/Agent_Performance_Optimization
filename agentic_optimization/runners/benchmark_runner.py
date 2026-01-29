"""
Benchmark Runner
Executes benchmarks and collects performance data
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import psutil

logger = logging.getLogger('BenchmarkRunner')


class BenchmarkRunner:
    """Runs benchmarks and profiles code"""
    
    def run(self, command: str, repo_path: Path, timeout: int = 60) -> Dict[str, Any]:
        """
        Run benchmark command
        
        Args:
            command: Shell command to execute
            repo_path: Repository path
            timeout: Timeout in seconds
        
        Returns:
            Execution results
        """
        logger.info(f"Running: {command}")
        
        start_time = time.perf_counter()
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor process
            try:
                proc = psutil.Process(process.pid)
                max_memory = 0
                cpu_samples = []
                
                while process.poll() is None:
                    try:
                        mem_mb = proc.memory_info().rss / 1024 / 1024
                        cpu_pct = proc.cpu_percent(interval=0.1)
                        max_memory = max(max_memory, mem_mb)
                        cpu_samples.append(cpu_pct)
                    except psutil.NoSuchProcess:
                        break
                
                avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            except:
                max_memory = 0
                avg_cpu = 0
            
            stdout, stderr = process.communicate(timeout=timeout)
            elapsed = time.perf_counter() - start_time
            
            return {
                'status': 'success' if process.returncode == 0 else 'failed',
                'returncode': process.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': elapsed,
                'memory_mb': max_memory,
                'cpu_percent': avg_cpu
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"Command timed out after {timeout}s")
            return {
                'status': 'timeout',
                'returncode': -1,
                'stdout': '',
                'stderr': f'Timeout after {timeout}s',
                'execution_time': timeout,
                'memory_mb': 0,
                'cpu_percent': 0
            }
        except Exception as e:
            logger.error(f"Benchmark error: {e}", exc_info=True)
            return {
                'status': 'error',
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0,
                'memory_mb': 0,
                'cpu_percent': 0
            }
    
    def profile(self, code_path: Path, input_size: int) -> Dict[str, Any]:
        """
        Profile code execution
        
        Args:
            code_path: Path to Python file
            input_size: Input size for profiling
        
        Returns:
            Profiling metrics
        """
        logger.info(f"Profiling {code_path} with size {input_size}")
        
        command = f"python {code_path} --size {input_size}"
        result = self.run(command, code_path.parent, timeout=30)
        
        return {
            'execution_time': result['execution_time'],
            'memory_mb': result['memory_mb'],
            'cpu_percent': result['cpu_percent'],
            'success': result['status'] == 'success',
            'output': result.get('stdout', '')
        }