import os
import sys
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows console
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    root_dir = Path(__file__).resolve().parent
    app_path = root_dir / "src" / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)

    print("==================================================")
    print("Launching TerraScan - Smart Waste Detection...")
    print(f"   Target: {app_path}")
    print("   Open browser at: http://localhost:8501")
    print("==================================================")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=str(root_dir),
        env=env
    )

if __name__ == "__main__":
    main()
