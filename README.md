# Digital-Twin-suite
# Operational Infrastructure Structural Health Monitoring (SHM) Suite

Unified structural digital twin and deep-learning early warning system for real-time post-tensioned bridge infrastructure monitoring.

## System Capabilities
* **3D Structural Digital Twin:** Real-time spatial surrogate mapping 100 Hz triaxial telemetry to full-field stress and strain tensors across 600 structural nodes and 15 post-tensioning tendons (285 elements / 300 discrete nodes).
* **Early Warning System (EWS):** Unsupervised 1D-CNN autoencoder evaluating reconstruction MSE and continuous Anomaly Index ($AI$) against a $3\sigma$ operational baseline, with automated forensic snapshots at $AI \ge 2.0$.
* **Dual Execution Modes:** Runs as standalone desktop PyQt dashboards or browser-accessible applications via GitHub Codespaces (`desktop-lite`).

## Quickstart

### Prerequisites
* Python 3.10+
* Git LFS (`git lfs install`)

### Local Installation
\`\`\`bash
git clone https://github.com/<ORG>/<REPO>.git
cd <REPO>
pip install -r requirements.txt
\`\`\`

### Execution
* **Launch 3D Digital Twin:** \`python apps/mthl_digital_twin_gui.py\`
* **Launch Early Warning System:** \`python apps/mthl_ews_gui.py\`
* **Launch Web Control Server:** \`python web/shm_server.py\` (Browse to \`http://localhost:8050\`)
