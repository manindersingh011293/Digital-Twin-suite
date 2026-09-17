"""
GENERIC STRUCTURAL HEALTH MONITORING (SHM) PORTAL SERVER
File: shm_server.py
Binds: 0.0.0.0:8050 (Accessible locally and across LAN)
"""

import os
import sys
import subprocess
from flask import Flask, Response

app = Flask(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DIGITAL_TWIN_SCRIPT = "mthl_digital_twin_gui.py"
EWS_SCRIPT = "mthl_ews_gui.py"

@app.route('/')
def portal_home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype='text/html')
    return "<h3>Error: index.html not found in the project root directory.</h3>", 404

@app.route('/digital-twin')
def launch_digital_twin():
    if os.path.exists(DIGITAL_TWIN_SCRIPT):
        subprocess.Popen([sys.executable, DIGITAL_TWIN_SCRIPT])
        msg = "Launching 3D Structural Digital Twin..."
    else:
        msg = f"Error: Target script '{DIGITAL_TWIN_SCRIPT}' not found."
    return f"""
    <html><body style="font-family:sans-serif; text-align:center; padding-top:60px;">
      <h2>{msg}</h2>
      <p>The desktop application process has been dispatched. You can close this window.</p>
    </body></html>
    """

@app.route('/early-warning')
def launch_ews():
    if os.path.exists(EWS_SCRIPT):
        subprocess.Popen([sys.executable, EWS_SCRIPT])
        msg = "Launching Real-Time Early Warning System..."
    else:
        msg = f"Error: Target script '{EWS_SCRIPT}' not found."
    return f"""
    <html><body style="font-family:sans-serif; text-align:center; padding-top:60px;">
      <h2>{msg}</h2>
      <p>The desktop application process has been dispatched. You can close this window.</p>
    </body></html>
    """

if __name__ == "__main__":
    print("=" * 60)
    print(" SHM MONITORING PORTAL RUNNING: http://0.0.0.0:8050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8050, debug=False)