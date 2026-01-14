import time, os
from pathlib import Path

# Get all the test file paths
root = Path(".")
paths = root.rglob("*.py")

EXCLUDE = {"run.py", "__pycache__", ".git"}

start = time.perf_counter()
for path in paths:
    if any(part in EXCLUDE for part in path.parts):
        continue
    print(f"\nRunning {str(path)}")
    os.system(f"pytest -q {str(path)}")
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
