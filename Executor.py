# run_sequential_subprocess.py
import sys
import subprocess
from pathlib import Path

# List the scripts to run in order
SCRIPTS = [
    "script1.py",
    "script2.py",
    "script3.py",
]

def run_scripts(scripts, python_exe=None, cwd=None, timeout=None):
    python = python_exe or sys.executable
    for script in scripts:
        path = Path(script)
        if not path.exists():
            print(f"SKIP: {script} not found")
            continue

        print(f"RUNNING: {script}")
        result = subprocess.run(
            [python, str(path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            print(f"ERROR: {script} exited with code {result.returncode}")
            return result.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(run_scripts(SCRIPTS))
