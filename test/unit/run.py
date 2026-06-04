import time, os, sys
import subprocess
from pathlib import Path

# Get all the test file paths
root = Path(".")
paths = root.rglob("*.py")

EXCLUDE = {"run.py", "__pycache__", ".git", "conftest.py", "__init__.py"}
EXCLUDE_PREFIX = {"make_test_"}

failed = False

start = time.perf_counter()
for path in paths:
    # Skip exact matches
    if any(part in EXCLUDE for part in path.parts):
        continue

    # Skip if any part starts with any prefix in EXCLUDE_PREFIX
    PREFIX_TUPLE = tuple(EXCLUDE_PREFIX)
    if any(part.startswith(PREFIX_TUPLE) for part in path.parts):
        continue

    print(f"\nRunning {str(path)}")
    sys.stdout.flush()
    result = subprocess.run(["pytest", "-q", str(path)])
    if result.returncode != 0:
        failed = True  # makes sure all tests are run

end = time.perf_counter()
total_time = end - start

if total_time >= 24 * 60 * 60:
    total_time = total_time / 24 / 60 / 60
    print(f"\nTotal unit test runtime: {total_time:.3f} days")
elif total_time >= 60 * 60:
    total_time = total_time / 60 / 60
    print(f"\nTotal unit test runtime: {total_time:.3f} hours")
elif total_time >= 60:
    total_time = total_time / 60
    print(f"\nTotal unit test runtime: {total_time:.3f} minutes")
else:
    print(f"\nTotal unit test runtime: {total_time:.3f} seconds")

if failed:
    sys.exit(1)
