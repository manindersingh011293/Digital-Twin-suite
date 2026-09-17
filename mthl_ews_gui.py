"""
STRUCTURAL HEALTH MONITORING: REAL-TIME EARLY WARNING SYSTEM (EWS)
File: mthl_ews_gui.py
Theme: High-Contrast Light Engineering
"""

import os
import sys
import time
import math
import threading
from collections import deque
import numpy as np
import torch
import torch.nn as nn

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mthl_theme import THEME, app_qss, btn_style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"

class Conv1DVerticalAutoencoder(nn.Module):
    def __init__(self, in_channels=1, window_size=256):
        super(Conv1DVerticalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(16, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def classify_health_event(w_data, ai, thresh_ambient=0.015):
    std_amp = float(np.std(w_data))
    max_amp = float(np.max(np.abs(w_data)))
    is_ambient = (std_amp < thresh_ambient) and (max_amp < 0.05)

    if is_ambient:
        if ai < 1.0: return "HEALTHY AMBIENT", "#059669", "#D1FAE5", 0
        else: return "STRUCTURAL DEGRADATION\n(BASELINE SHIFT)", "#7C3AED", "#EDE9FE", 4
    else:
        if ai < 1.0: return "NOMINAL TRAFFIC", "#0284C7", "#E0F2FE", 1
        elif ai < 2.0: return "TRANSIENT WARNING\n(HEAVY LOAD)", "#D97706", "#FEF3C7", 2
        else: return "CRITICAL OVERLOAD\n(ANOMALY)", "#DC2626", "#FEE2E2", 3

class EWSBackendEngine:
    def __init__(self, telemetry_path="20260106_Acc_100Hz.txt", model_path="best_ae_vertical_model.pth"):
        self.telemetry_path, self.model_path = telemetry_path, model_path
        self.window_size, self.history_len = 256, 1000
        self.running, self.playback_speed = False, 1.0
        self.stop_event, self.lock = threading.Event(), threading.Lock()

        self.ae_model = Conv1DVerticalAutoencoder(1, self.window_size).to(device)
        self.ae_model.eval()
        self.stats = {"scaler_mean": 0.0, "scaler_scale": 1.0, "thresh_warning": 0.045}
        self._load_checkpoint()

        self.sps = 100.0
        self.dt = 1.0 / self.sps
        self.sample_idx = 0
        self.file_obj = None
        self.data_start_offset = 0
        self._init_file_feeder()

        self.time_buffer = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.ay_buffer = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.ai_buffer = deque([0.0] * self.history_len, maxlen=self.history_len)

        self.fft_freqs = np.fft.rfftfreq(self.window_size, d=self.dt)
        self.fft_window_fn = np.hanning(self.window_size)

        self.live_state = {
            "timestamp": time.time(), "state_version": 0, "telemetry_time": 0.0, "latency_ms": 0.0,
            "curr_ay": 0.0, "rms_ay": 0.0, "live_mse": 0.0, "live_ai": 0.0, "peak_ai": 0.0,
            "category": "INITIALIZING", "fg_color": "#0284C7", "bg_color": "#E0F2FE", "severity": 0,
            "time_history": np.zeros(self.history_len), "ay_history": np.zeros(self.history_len), "ai_history": np.zeros(self.history_len),
            "ae_time": np.linspace(-2.56, 0.0, self.window_size), "ae_raw": np.zeros(self.window_size), "ae_recon": np.zeros(self.window_size),
            "fft_mag_db": np.full_like(self.fft_freqs, -80.0), "new_event": None,
        }

    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location=device, weights_only=False)
                self.ae_model.load_state_dict(ckpt["model_state_dict"])
                s = ckpt.get("baseline_stats", {})
                self.stats["scaler_mean"] = s.get("scaler_mean", [0.0])[0]
                self.stats["scaler_scale"] = s.get("scaler_scale", [1.0])[0]
                self.stats["thresh_warning"] = s.get("thresh_3sigma", s.get("mean_mse", 0.012) + 3.0 * s.get("std_mse", 0.011))
            except Exception as e:
                print(f"[WARN] Error loading AE checkpoint: {e}")

    def _init_file_feeder(self):
        if os.path.exists(self.telemetry_path):
            self.file_obj = open(self.telemetry_path, "r", encoding="utf-8", errors="ignore")
            line = self.file_obj.readline()
            while line:
                if line.startswith("SPS"):
                    try: self.sps = float(line.split("=")[1].strip()); self.dt = 1.0 / self.sps
                    except Exception: pass
                elif "----------**********----------" in line:
                    self.data_start_offset = self.file_obj.tell(); break
                line = self.file_obj.readline()

    def _read_sample(self):
        if self.file_obj is None:
            t = self.sample_idx * self.dt
            traffic = 0.35 * math.sin(2.0 * math.pi * 1.85 * t) if (int(t) % 15 < 4) else 0.0
            impact = 1.2 * math.exp(-((t % 25 - 5)**2) / 0.1)
            ay = 0.04 * math.sin(2.0 * math.pi * 2.83 * t) + traffic + impact + np.random.normal(0, 0.005)
            self.sample_idx += 1
            return t, ay

        line = self.file_obj.readline()
        if not line:
            self.file_obj.seek(self.data_start_offset)
            self.sample_idx = 0
            line = self.file_obj.readline()
            if not line: return 0.0, 0.0

        tokens = line.replace(",", " ").replace("\t", " ").split()
        if len(tokens) >= 4:
            self.sample_idx += 1
            return float(tokens[0]), float(tokens[1])
        return self.sample_idx * self.dt, 0.0

    def _worker_loop(self):
        next_tick = time.perf_counter()
        in_alert_latch = False

        while self.running:
            t_curr, ay_curr = self._read_sample()
            with self.lock:
                self.time_buffer.append(t_curr)
                self.ay_buffer.append(ay_curr)

            if len(self.ay_buffer) >= self.window_size:
                w_raw = np.array(list(self.ay_buffer)[-self.window_size:], dtype=np.float32)
                t0 = time.perf_counter()
                x_std = (w_raw - self.stats["scaler_mean"]) / (self.stats["scaler_scale"] + 1e-7)
                x_t = torch.from_numpy(x_std).unsqueeze(0).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    recon_t = self.ae_model(x_t)
                    recon_np = recon_t.squeeze().cpu().numpy()
                    mse = float(np.mean((recon_np - x_std) ** 2))

                latency = (time.perf_counter() - t0) * 1000.0
                ai_val = mse / (self.stats["thresh_warning"] + 1e-7)

                with self.lock:
                    self.ai_buffer.append(ai_val)

                cat, fg, bg, sev = classify_health_event(w_raw, ai_val)
                spec = np.abs(np.fft.rfft(w_raw * self.fft_window_fn)) / (self.window_size * 0.5)
                fft_db = (20.0 * np.log10(spec + 1e-9)).astype(np.float32)

                new_event = None
                if ai_val >= 1.0:
                    if not in_alert_latch:
                        in_alert_latch = True
                        new_event = {
                            "timestamp": f"{t_curr:.2f}s", "category": cat.replace("\n", " "),
                            "ai": f"{ai_val:.2f}", "peak_acc": f"{np.max(np.abs(w_raw)):.3f}",
                            "severity": sev, "raw": w_raw, "recon": recon_np,
                            "t_win": np.array(list(self.time_buffer)[-self.window_size:])
                        }
                else:
                    in_alert_latch = False

                with self.lock:
                    self.live_state.update({
                        "timestamp": time.time(), "state_version": self.live_state["state_version"] + 1,
                        "telemetry_time": t_curr, "latency_ms": latency, "curr_ay": ay_curr,
                        "rms_ay": float(np.sqrt(np.mean(w_raw**2))), "live_mse": mse, "live_ai": ai_val,
                        "peak_ai": max(self.live_state["peak_ai"], ai_val), "category": cat,
                        "fg_color": fg, "bg_color": bg, "severity": sev,
                        "time_history": np.array(self.time_buffer), "ay_history": np.array(self.ay_buffer),
                        "ai_history": np.array(self.ai_buffer), "ae_raw": x_std, "ae_recon": recon_np,
                        "fft_mag_db": fft_db
                    })
                    if new_event is not None:
                        self.live_state["new_event"] = new_event

            next_tick += self.dt / max(0.01, self.playback_speed)
            if self.stop_event.wait(max(0.0, next_tick - time.perf_counter())):
                break

    def start(self):
        self.running = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        self.stop_event.set()

class EWSScopeCanvas(FigureCanvas):
    def __init__(self, parent=None, width=13, height=8, dpi=105):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="#ffffff")
        super().__init__(self.fig)
        self.setParent(parent)

        gs = gridspec.GridSpec(2, 2, figure=self.fig, left=0.06, right=0.97, top=0.93, bottom=0.08, hspace=0.35, wspace=0.20)

        def style(ax, title, ylabel, xlabel):
            ax.set_facecolor(THEME["panel_2"])
            ax.tick_params(colors=THEME["text_dim"], labelsize=8)
            ax.set_title(title, color=THEME["text"], fontsize=10, fontweight="bold", pad=5)
            ax.set_ylabel(ylabel, color=THEME["text_dim"], fontsize=8)
            ax.set_xlabel(xlabel, color=THEME["text_dim"], fontsize=8)
            ax.grid(True, linestyle="--", color=THEME["grid"], alpha=0.9)
            for s in ax.spines.values(): s.set_color(THEME["border"])

        self.ax_telemetry = self.fig.add_subplot(gs[0, 0])
        style(self.ax_telemetry, "1. Rolling Accelerometer Telemetry (ay)", "Accel (m/s²)", "Time (s)")
        self.line_raw, = self.ax_telemetry.plot([], [], color=THEME["info"], lw=1.2)
        self.ax_telemetry.set_ylim(-2.0, 2.0)

        self.ax_ae = self.fig.add_subplot(gs[0, 1])
        style(self.ax_ae, "2. Autoencoder Reconstruction & Dynamic MSE Gap", "Normalized Accel", "Window (s)")
        self.t_ae = np.linspace(-2.56, 0.0, 256)
        self.line_ae_raw, = self.ax_ae.plot(self.t_ae, np.zeros(256), color=THEME["info"], lw=1.5, label="Input Signal")
        self.line_ae_rec, = self.ax_ae.plot(self.t_ae, np.zeros(256), color="#0f172a", linestyle="--", lw=1.2, label="AE Reconstructed")
        self.error_fill = None
        self.ax_ae.set_xlim(-2.56, 0.0)
        self.ax_ae.set_ylim(-3.5, 3.5)
        self.ax_ae.legend(loc="upper left", fontsize=7.5, facecolor="#ffffff", edgecolor=THEME["border"])

        self.ax_ai = self.fig.add_subplot(gs[1, 0])
        style(self.ax_ai, "3. Anomaly Index (AI) Trajectory", "Anomaly Index", "Time (s)")
        self.line_ai, = self.ax_ai.plot([], [], color=THEME["text"], lw=1.5)
        self.ax_ai.axhline(1.0, color=THEME["warn"], linestyle="--", lw=1.2, label="Warning (AI=1.0)")
        self.ax_ai.axhline(2.0, color=THEME["crit"], linestyle="--", lw=1.2, label="Critical (AI=2.0)")
        self.pt_ai = self.ax_ai.scatter([0], [0], color=THEME["ok"], s=70, edgecolors='white', lw=1.5, zorder=10)
        self.ax_ai.set_ylim(0.0, 6.0)
        self.ax_ai.legend(loc="upper left", fontsize=7.5, facecolor="#ffffff", edgecolor=THEME["border"])

        self.ax_fft = self.fig.add_subplot(gs[1, 1])
        style(self.ax_fft, "4. Vertical Frequency Spectrum (0-50 Hz)", "Power (dB)", "Frequency (Hz)")
        self.line_fft, = self.ax_fft.plot([], [], color=THEME["violet"], lw=1.2)
        self.ax_fft.set_xlim(0.0, 50.0)
        self.ax_fft.set_ylim(-80.0, 20.0)

class MTHLEWSMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Early Warning System (EWS) — Anomaly Identification & Degradation")
        self.resize(1920, 1080)
        self.setStyleSheet(app_qss())

        self.backend = EWSBackendEngine()
        self._build_ui()
        self.backend.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._render_frame)
        self.timer.start(33)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._build_header())

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(10)
        self.canvas = EWSScopeCanvas(self)
        main_layout.addWidget(self.canvas, 3)
        main_layout.addWidget(self._build_sidebar(), 1)
        root.addLayout(main_layout)

    def _build_header(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 6px;")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(15)

        title = QtWidgets.QLabel("STRUCTURAL EARLY WARNING SYSTEM (EWS)")
        title.setStyleSheet(f"color: {THEME['text']}; font-weight: bold; font-size: 12px;")
        h.addWidget(title)
        h.addWidget(self._vline())

        self.lbl_hw = QtWidgets.QLabel(f"DEVICE: {hardware_name}")
        self.lbl_hw.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold; font-size: 10px;")
        h.addWidget(self.lbl_hw)
        h.addWidget(self._vline())

        self.lbl_latency = QtWidgets.QLabel("INFERENCE: -- ms")
        self.lbl_latency.setStyleSheet(f"color: {THEME['text_dim']}; font-weight: bold; font-size: 10px;")
        h.addWidget(self.lbl_latency)

        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.lbl_fps.setStyleSheet(f"color: {THEME['text_dim']}; font-weight: bold; font-size: 10px;")
        h.addWidget(self.lbl_fps)
        h.addStretch()

        self.lbl_clock = QtWidgets.QLabel("TELEMETRY T: 0.00 s")
        self.lbl_clock.setStyleSheet(f"color: {THEME['violet']}; font-family: Consolas; font-weight: bold; font-size: 11px;")
        h.addWidget(self.lbl_clock)
        h.addWidget(self._vline())

        self.lbl_status = QtWidgets.QLabel("LIVE MONITORING")
        self.lbl_status.setStyleSheet(f"color: white; background-color: {THEME['ok']}; font-weight: bold; font-size: 10px; border-radius: 3px; padding: 4px 8px;")
        h.addWidget(self.lbl_status)
        return bar

    def _build_sidebar(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setFixedWidth(420)
        scroll.setWidgetResizable(True)

        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)

        lbl_cat_head = QtWidgets.QLabel("LIVE STRUCTURAL STATE")
        lbl_cat_head.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 9px; font-weight: bold; letter-spacing: 1px;")
        v.addWidget(lbl_cat_head)

        self.status_badge = QtWidgets.QLabel("INITIALIZING...")
        self.status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setFixedHeight(85)
        self.status_badge.setStyleSheet(f"QLabel {{ color: #059669; background-color: #D1FAE5; font-size: 16px; font-weight: 800; border: 3px solid #059669; border-radius: 8px; padding: 6px; }}")
        v.addWidget(self.status_badge)

        # Metrics Card
        metrics_group = QtWidgets.QGroupBox("REAL-TIME DIAGNOSTIC METRICS")
        mg = QtWidgets.QGridLayout(metrics_group)
        mg.setContentsMargins(10, 10, 10, 10)
        mg.setSpacing(6)

        def _met_row(row, title, default="--"):
            tl = QtWidgets.QLabel(title)
            tl.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 9.5px;")
            vl = QtWidgets.QLabel(default)
            vl.setStyleSheet(f"color: {THEME['text']}; font-family: Consolas; font-weight: bold; font-size: 11px;")
            mg.addWidget(tl, row, 0)
            mg.addWidget(vl, row, 1, QtCore.Qt.AlignmentFlag.AlignRight)
            return vl

        self.val_ai = _met_row(0, "Anomaly Index (AI):")
        self.val_peak_ai = _met_row(1, "Session Peak AI:")
        self.val_mse = _met_row(2, "Live Recon MSE:")
        self.val_thresh = _met_row(3, "3σ Baseline Limit:", f"{self.backend.stats['thresh_warning']:.5f}")
        self.val_rms = _met_row(4, "RMS Acceleration:")
        v.addWidget(metrics_group)

        # Event Log
        log_group = QtWidgets.QGroupBox("FORENSIC ANOMALY EVENT LOG")
        lv = QtWidgets.QVBoxLayout(log_group)
        lv.setContentsMargins(6, 8, 6, 6)

        self.table_log = QtWidgets.QTableWidget(0, 4)
        self.table_log.setHorizontalHeaderLabels(["Time", "Category", "Peak AI", "Acc Max"])
        self.table_log.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_log.setFixedHeight(180)
        lv.addWidget(self.table_log)

        btn_clear_table = QtWidgets.QPushButton("Clear Log")
        btn_clear_table.clicked.connect(lambda: self.table_log.setRowCount(0))
        lv.addWidget(btn_clear_table)
        v.addWidget(log_group)

        # 5-State Legend
        legend_group = QtWidgets.QGroupBox("5-STATE HEALTH MATRIX")
        lg = QtWidgets.QVBoxLayout(legend_group)
        lg.setSpacing(4)
        for label, col in [
            ("Healthy Ambient (Low Amp, Low AI)", "#059669"),
            ("Nominal Traffic (Mid Amp, Low AI)", "#0284C7"),
            ("Transient Warning (Heavy Axle, AI 1-2)", "#D97706"),
            ("Critical Overload (Dynamic Overload, AI > 2)", "#DC2626"),
            ("Structural Degradation (Baseline Shift)", "#7C3AED")
        ]:
            row = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet(f"color: {col}; font-size: 14px;")
            txt = QtWidgets.QLabel(label)
            txt.setStyleSheet(f"font-size: 9px; color: {THEME['text']};")
            row.addWidget(dot)
            row.addWidget(txt)
            row.addStretch()
            lg.addLayout(row)
        v.addWidget(legend_group)

        # Controls
        ctrl_group = QtWidgets.QGroupBox("STREAM CONTROL")
        cg = QtWidgets.QVBoxLayout(ctrl_group)
        cg.setSpacing(6)

        sp_row = QtWidgets.QHBoxLayout()
        sp_lbl = QtWidgets.QLabel("Rate:")
        sp_lbl.setStyleSheet(f"font-size: 9.5px; color: {THEME['text_dim']};")
        self.combo_speed = QtWidgets.QComboBox()
        self.combo_speed.addItems(["1× (100 Hz)", "2×", "5×", "10×"])
        self.combo_speed.currentIndexChanged.connect(lambda idx: setattr(self.backend, 'playback_speed', [1.0, 2.0, 5.0, 10.0][idx]))
        sp_row.addWidget(sp_lbl)
        sp_row.addWidget(self.combo_speed)
        cg.addLayout(sp_row)

        self.btn_pause = QtWidgets.QPushButton("Pause Stream")
        self.btn_pause.setStyleSheet(btn_style(THEME["accent"]))
        self.btn_pause.clicked.connect(self._toggle_stream)
        cg.addWidget(self.btn_pause)

        self.chk_autodump = QtWidgets.QCheckBox("Auto-Dump .npz on AI >= 2.0")
        self.chk_autodump.setChecked(True)
        cg.addWidget(self.chk_autodump)
        v.addWidget(ctrl_group)

        v.addStretch()
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _vline():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setStyleSheet(f"color: {THEME['border']}; background: {THEME['border']};")
        line.setFixedWidth(1)
        return line

    def _render_frame(self):
        with self.backend.lock:
            st = dict(self.backend.live_state)

        t_hist = st["time_history"]
        ay_hist = st["ay_history"]
        ai_hist = st["ai_history"]

        if len(t_hist) < 2: return

        self.canvas.line_raw.set_data(t_hist, ay_hist)
        self.canvas.ax_telemetry.set_xlim(t_hist[0], t_hist[-1])
        pk = max(0.5, float(np.max(np.abs(ay_hist))) * 1.15)
        self.canvas.ax_telemetry.set_ylim(-pk, pk)

        x_raw = st["ae_raw"]
        x_rec = st["ae_recon"]
        self.canvas.line_ae_raw.set_ydata(x_raw)
        self.canvas.line_ae_rec.set_ydata(x_rec)

        if self.canvas.error_fill is not None:
            self.canvas.error_fill.remove()
        self.canvas.error_fill = self.canvas.ax_ae.fill_between(
            self.canvas.t_ae, x_raw, x_rec, color="#dc2626", alpha=0.35
        )
        pk_norm = max(3.0, float(np.max(np.abs(x_raw))) * 1.2)
        self.canvas.ax_ae.set_ylim(-pk_norm, pk_norm)

        self.canvas.line_ai.set_data(t_hist, ai_hist)
        self.canvas.ax_ai.set_xlim(t_hist[0], t_hist[-1])
        cur_ai = st["live_ai"]
        self.canvas.pt_ai.set_offsets([[t_hist[-1], cur_ai]])
        self.canvas.pt_ai.set_facecolor(st["fg_color"])
        self.canvas.ax_ai.set_ylim(0.0, max(3.5, float(np.max(ai_hist)) * 1.15))

        self.canvas.line_fft.set_data(self.backend.fft_freqs, st["fft_mag_db"])
        self.canvas.draw_idle()

        cat = st["category"]
        fg = st["fg_color"]
        bg = st["bg_color"]
        self.status_badge.setText(cat)
        self.status_badge.setStyleSheet(f"QLabel {{ color: {fg}; background-color: {bg}; font-size: 15px; font-weight: 800; border: 3px solid {fg}; border-radius: 8px; padding: 6px; }}")

        self.val_ai.setText(f"{cur_ai:.2f}")
        self.val_peak_ai.setText(f"{st['peak_ai']:.2f}")
        self.val_mse.setText(f"{st['live_mse']:.5f}")
        self.val_rms.setText(f"{st['rms_ay']:.4f} m/s²")
        self.lbl_clock.setText(f"TELEMETRY T: {st['telemetry_time']:8.2f} s")
        self.lbl_latency.setText(f"INFERENCE: {st['latency_ms']:4.1f} ms")

        new_ev = st.get("new_event", None)
        if new_ev is not None:
            self.backend.live_state["new_event"] = None
            self._log_event_row(new_ev)

    def _log_event_row(self, ev):
        row = self.table_log.rowCount()
        self.table_log.insertRow(row)

        item_t = QtWidgets.QTableWidgetItem(ev["timestamp"])
        item_cat = QtWidgets.QTableWidgetItem(ev["category"])
        item_ai = QtWidgets.QTableWidgetItem(ev["ai"])
        item_pk = QtWidgets.QTableWidgetItem(ev["peak_acc"])

        if float(ev["ai"]) >= 2.0:
            for item in (item_t, item_cat, item_ai, item_pk):
                item.setForeground(QtGui.QColor("#dc2626"))
                item.setBackground(QtGui.QColor("#fee2e2"))
        elif float(ev["ai"]) >= 1.0:
            for item in (item_t, item_cat, item_ai, item_pk):
                item.setForeground(QtGui.QColor("#d97706"))

        self.table_log.setItem(row, 0, item_t)
        self.table_log.setItem(row, 1, item_cat)
        self.table_log.setItem(row, 2, item_ai)
        self.table_log.setItem(row, 3, item_pk)
        self.table_log.scrollToBottom()

        if float(ev["ai"]) >= 2.0 and self.chk_autodump.isChecked():
            os.makedirs("forensic_dumps", exist_ok=True)
            fpath = f"forensic_dumps/CRITICAL_t{float(ev['timestamp'][:-1]):.2f}_AI{ev['ai']}.npz"
            np.savez_compressed(
                fpath, time_sec=ev["t_win"], accel_raw=ev["raw"],
                accel_recon=ev["recon"], anomaly_index=float(ev["ai"])
            )

    def _toggle_stream(self):
        if self.backend.running:
            self.backend.stop()
            self.btn_pause.setText("Resume Stream")
            self.btn_pause.setStyleSheet(btn_style(THEME["ok"]))
            self.lbl_status.setText("STREAM PAUSED")
            self.lbl_status.setStyleSheet(f"color: white; background-color: {THEME['warn']}; font-weight: bold; font-size: 10px; border-radius: 3px; padding: 4px 8px;")
        else:
            self.backend.start()
            self.btn_pause.setText("Pause Stream")
            self.btn_pause.setStyleSheet(btn_style(THEME["accent"]))
            self.lbl_status.setText("LIVE MONITORING")
            self.lbl_status.setStyleSheet(f"color: white; background-color: {THEME['ok']}; font-weight: bold; font-size: 10px; border-radius: 3px; padding: 4px 8px;")

    def closeEvent(self, event):
        self.backend.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = MTHLEWSMainWindow()
    gui.show()
    sys.exit(app.exec())