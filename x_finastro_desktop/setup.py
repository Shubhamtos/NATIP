import subprocess
import sys

print("Installing Python packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("Installing Playwright Chromium...")
subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

print("\nSetup complete. Run: python app.py")
