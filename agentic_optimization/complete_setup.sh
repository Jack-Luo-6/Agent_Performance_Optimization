#!/bin/bash
# Complete setup script for refactored architecture

set -e

echo "======================================================================"
echo "Performance Optimization Architecture - Complete Setup"
echo "======================================================================"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p agents
mkdir -p mcp
mkdir -p opencode  
mkdir -p runners
mkdir -p artifacts/workloads
mkdir -p artifacts/optimizations
mkdir -p reference_docs
mkdir -p test_repo

echo "✓ Directories created"

# Check Python
echo ""
echo "Checking Python..."
python3 --version
echo "✓ Python OK"

# Create virtual environment
echo ""
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate and install
echo ""
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install openai psutil sweagent

echo "✓ Dependencies installed"

# Check for API key
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Check for mini_swe_agent
echo ""
if [ -d "mini_swe_agent" ]; then
    echo "✓ mini_swe_agent found"
else
    echo "⚠️  mini_swe_agent not found (optional)"
    echo "   System will use direct OpenAI calls"
fi

# Create example reference doc
echo ""
echo "Creating example reference documentation..."
cat > reference_docs/example_code_structure.md << 'EOF'
# Example Code Structure Reference

## Main Functions

### process_data(items: list) -> list
Processes a list of items through transformation pipeline.
- Input: List of items (any type)
- Output: Processed list
- Performance: O(n) expected

### find_duplicates(items: list) -> list
Finds duplicate items in a list.
- Input: List of items
- Output: List of duplicates
- Performance: Currently O(n²), target O(n)

### compute_statistics(numbers: list) -> dict
Computes basic statistics on numbers.
- Input: List of numbers
- Output: Dict with mean, min, max, variance
- Performance: O(n) with multiple passes

## Expected Usage

```python
data = [1, 2, 3, 4, 5]
processed = process_data(data)
duplicates = find_duplicates(data)
stats = compute_statistics(data)
```

## Performance Targets

- Handle inputs up to 100,000 elements
- Sub-second processing for typical cases
- Memory efficient (< 100MB for large inputs)
EOF

echo "✓ Reference documentation created"

# Create test repository
echo ""
echo "Creating test repository..."
cat > test_repo/main.py << 'EOF'
#!/usr/bin/env python3
"""Example target code - intentionally inefficient"""
import sys
import argparse

def find_duplicates(items):
    """O(n²) implementation"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def process_data(data):
    """Multiple inefficiencies"""
    result = []
    for item in data:
        if item not in result:
            result.append(item)
    
    # Bubble sort
    for i in range(len(result)):
        for j in range(len(result) - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result

def compute_statistics(numbers):
    """Multiple passes"""
    if not numbers:
        return {'mean': 0, 'variance': 0, 'min': 0, 'max': 0}
    
    total = sum(numbers)
    mean = total / len(numbers)
    variance_sum = sum((num - mean) ** 2 for num in numbers)
    variance = variance_sum / len(numbers)
    
    return {
        'mean': mean,
        'variance': variance,
        'min': min(numbers),
        'max': max(numbers)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000)
    args = parser.parse_args()
    
    data = list(range(args.size)) + list(range(args.size // 2))
    
    duplicates = find_duplicates(data[:min(100, len(data))])
    processed = process_data(data)
    stats = compute_statistics(processed)
    
    print(f"Processed {len(data)} items")
    return 0

if __name__ == '__main__':
    sys.exit(main())
EOF

chmod +x test_repo/main.py

cd test_repo
git init 2>/dev/null || true
git add main.py 2>/dev/null || true
git commit -m "Initial inefficient code" 2>/dev/null || true
cd ..

echo "✓ Test repository created"

# Create __init__.py files
echo ""
echo "Creating __init__.py files..."
touch agents/__init__.py
touch mcp/__init__.py
touch opencode/__init__.py
touch runners/__init__.py

echo "✓ Python packages initialized"

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Directory structure:"
echo "  agents/          - Agent logic (reasoning)"
echo "  mcp/             - MCP server (tools)"
echo "  opencode/        - Repository abstraction"
echo "  runners/         - Execution infrastructure"
echo "  artifacts/       - Generated outputs"
echo "  reference_docs/  - Reference documentation"
echo "  test_repo/       - Example target code"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set API key (if not set):"
echo "   export OPENAI_API_KEY='your-key-here'"
echo ""
echo "3. Run optimization:"
echo "   python orchestrator.py --repo test_repo --reference reference_docs/example_code_structure.md --iterations 5"
echo ""
echo "4. Analyze results:"
echo "   python analyze_results.py --details"
echo ""
echo "======================================================================"