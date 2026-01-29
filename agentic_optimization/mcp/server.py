"""
MCP Server - Capability exposure layer
Provides isolated tool access for agents
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runners.benchmark_runner import BenchmarkRunner
from runners.workload_runner import WorkloadRunner

logger = logging.getLogger('MCPServer')


class MCPServer:
    """
    Model Context Protocol Server
    Exposes tools to agents with strict boundaries
    """
    
    def __init__(self):
        self.benchmark_runner = BenchmarkRunner()
        self.workload_runner = WorkloadRunner()
        logger.info("MCP Server initialized")
    
    # ========================================================================
    # WORKLOAD TOOLS (exposed to workload_agent)
    # ========================================================================
    
    def run_workload(self, workload_code: str, repo_path: Path, 
                     workload_type: str = "original") -> Dict[str, Any]:
        """
        Execute workload code against target repository
        
        Args:
            workload_code: Python code that generates workload
            repo_path: Path to target repository
            workload_type: "original", "human", or "optimized"
        
        Returns:
            Execution results with metrics
        """
        logger.info(f"Running workload (type: {workload_type}) on {repo_path}")
        
        try:
            result = self.workload_runner.run(
                workload_code=workload_code,
                repo_path=repo_path,
                workload_type=workload_type
            )
            logger.info(f"✓ Workload completed: {result.get('status', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Workload execution failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'metrics': self._error_metrics()
            }
    
    # ========================================================================
    # BENCHMARK TOOLS (exposed to optimizer_agent)
    # ========================================================================
    
    def run_benchmark(self, command: str, repo_path: Path, 
                     timeout: int = 60) -> Dict[str, Any]:
        """
        Run benchmark command in repository
        
        Args:
            command: Shell command to execute
            repo_path: Repository path
            timeout: Execution timeout in seconds
        
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark: {command}")
        
        try:
            result = self.benchmark_runner.run(
                command=command,
                repo_path=repo_path,
                timeout=timeout
            )
            logger.info(f"✓ Benchmark completed")
            return result
        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'stdout': '',
                'stderr': str(e)
            }
    
    def profile_code(self, code_path: Path, input_size: int) -> Dict[str, Any]:
        """
        Profile code execution
        
        Args:
            code_path: Path to code file
            input_size: Input size for profiling
        
        Returns:
            Profiling results
        """
        logger.info(f"Profiling {code_path} with size {input_size}")
        
        try:
            result = self.benchmark_runner.profile(
                code_path=code_path,
                input_size=input_size
            )
            return result
        except Exception as e:
            logger.error(f"Profiling failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'metrics': self._error_metrics()
            }
    
    # ========================================================================
    # SHARED UTILITIES
    # ========================================================================
    
    def _error_metrics(self) -> Dict[str, float]:
        """Return error metrics"""
        return {
            'execution_time': 999999.0,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'success': False
        }
    
    # ========================================================================
    # TOOL REGISTRY (for introspection)
    # ========================================================================
    
    def get_available_tools(self, agent_type: str) -> list[str]:
        """
        Get list of tools available to specific agent type
        
        Args:
            agent_type: "workload_agent" or "optimizer_agent"
        
        Returns:
            List of tool names
        """
        tools = {
            'workload_agent': [
                'run_workload',
            ],
            'optimizer_agent': [
                'run_benchmark',
                'profile_code',
            ]
        }
        return tools.get(agent_type, [])