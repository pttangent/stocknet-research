"""Launcher script for the Streamlit dashboard."""

import subprocess
import sys
import os


def main():
    """Launch the Streamlit dashboard."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "dashboard", "app.py")

    if not os.path.exists(app_path):
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    print("[LAUNCH] Starting Community Monitoring Radar Dashboard...")
    print(f"   App: {app_path}")
    print("   Press Ctrl+C to stop")
    print()

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
