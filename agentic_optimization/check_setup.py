#!/usr/bin/env python3
"""
Check if all required files are present for the architecture
"""

import sys
from pathlib import Path

def check_file(path, required=True):
    """Check if file exists"""
    p = Path(path)
    exists = p.exists()
    
    if exists:
        size = p.stat().st_size if p.is_file() else "DIR"
        print(f"  ✓ {path} ({size} bytes)" if isinstance(size, int) else f"  ✓ {path}")
        return True
    else:
        marker = "❌" if required else "⚠️ "
        print(f"  {marker} MISSING: {path}")
        return not required

def main():
    print("="*70)
    print("Checking Architecture Setup")
    print("="*70)
    
    all_good = True
    
    # Core files (REQUIRED)
    print("\n📁 Core Files (Required):")
    core_files = [
        "orchestrator.py",
        "mcp/server.py",
        "mcp/__init__.py",
        "opencode/repo.py",
        "opencode/__init__.py",
        "runners/benchmark_runner.py",
        "runners/workload_runner.py",
        "runners/__init__.py",
        "agents/workload_agent.py",
        "agents/optimizer_agent.py",
        "agents/__init__.py",
    ]
    
    for f in core_files:
        if not check_file(f, required=True):
            all_good = False
    
    # Support files
    print("\n📄 Support Files:")
    support_files = [
        "requirements.txt",
        "analyze_results.py",
    ]
    
    for f in support_files:
        check_file(f, required=False)
    
    # Directories
    print("\n📂 Directories:")
    dirs = [
        "artifacts",
        "artifacts/workloads",
        "artifacts/optimizations",
        "reference_docs",
        "test_repo",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        check_file(d, required=True)
    
    # Optional
    print("\n🔧 Optional:")
    check_file("mini_swe_agent", required=False)
    check_file("venv", required=False)
    
    # Check Python packages
    print("\n📦 Python Packages:")
    try:
        import openai
        print("  ✓ openai installed")
    except ImportError:
        print("  ❌ openai NOT installed - run: pip install openai")
        all_good = False
    
    try:
        import psutil
        print("  ✓ psutil installed")
    except ImportError:
        print("  ❌ psutil NOT installed - run: pip install psutil")
        all_good = False
    
    # Check environment
    print("\n🔑 Environment:")
    import os
    if os.getenv('OPENAI_API_KEY'):
        print("  ✓ OPENAI_API_KEY is set")
    else:
        print("  ⚠️  OPENAI_API_KEY not set (will use template mode)")
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✅ All required files present!")
        print("\nYou can run:")
        print("  python orchestrator.py --repo test_repo --iterations 3")
        print("\nOr test without API (uses templates):")
        print("  ./test_workflow.sh")
    else:
        print("❌ Some required files are missing!")
        print("\nCreate missing files or run:")
        print("  ./complete_setup.sh")
    print("="*70)
    
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())