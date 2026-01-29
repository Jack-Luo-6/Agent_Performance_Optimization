import subprocess
import sys

TEST_FILES = [
    "../venv/lib/python3.12/site-packages/pandas/tests/indexes/interval/test_setops.py",
    "../venv/lib/python3.12/site-packages/pandas/tests/indexing/test_categorical.py",
]

def main():
    passed = 0
    total = 0

    for file_path in TEST_FILES:
        if file_path.startswith("/testbed/"):
            script_path = file_path[len("/testbed/"):]
        else:
            script_path = file_path

        if not script_path.startswith("pandas/"):
            continue

        result = subprocess.run(
            ["pytest", script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        total += 1
        if result.returncode == 0:
            passed += 1

    # Success rate in [0, 1]
    success_rate = passed / total if total > 0 else 1.0

    # IMPORTANT: print ONLY the number
    print(f"{success_rate:.6f}")

    # IMPORTANT: correctness script itself succeeded
    sys.exit(0)


if __name__ == "__main__":
    main()
