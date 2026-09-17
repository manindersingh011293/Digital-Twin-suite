"""
STRUCTURAL HEALTH MONITORING: 3D DIGITAL TWIN DASHBOARD
File: mthl_digital_twin_gui.py
Theme: LIGHT
"""

import os
import sys
import time
from collections import deque
import numpy as np
import torch

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mthl_digital_twin import MTHLDigitalTwin, hardware_name
from mthl_theme import THEME, LEVEL_COLORS, app_qss, btn_style

SENSOR_NODE_ID  = 278
OPENSEES_SENSOR = 309
SENSOR_ARM      = 0.85          
MESH_ALPHA      = 0.04          
MESH_EDGE       = "#94a3b8"     

class RealisticBoxGirderModel:
    def __init__(self, span_length=59.818):
        self.L = span_length
        self.y_top, self.y_bot = 3.85, 0.00
        self.z_deck_left, self.z_deck_right = -8.00, 8.00
        self.z_web_top_L, self.z_web_top_R = -3.20, 3.20
        self.z_web_bot_L, self.z_web_bot_R = -2.43, 2.43

        L_pier, L_stitch, L_std = 2.000, 0.199, 3.260
        segment_lengths = [L_pier + L_stitch] + [L_std] * 17 + [L_stitch + L_pier]
        self.stations_x = np.array([0.0] + list(np.cumsum(segment_lengths)), dtype=np.float32)

        z_top = np.array([self.z_deck_left, -5.60, self.z_web_top_L, -1.60, 0.00, 1.60, self.z_web_top_R, 5.60, self.z_deck_right])
        z_bot = np.linspace(self.z_web_bot_L, self.z_web_bot_R, 5)
        y_web = np.linspace(self.y_bot, self.y_top, 5)

        X_top, Z_top = np.meshgrid(self.stations_x, z_top, indexing="ij")
        X_bot, Z_bot = np.meshgrid(self.stations_x, z_bot, indexing="ij")
        X_lw, Y_lw = np.meshgrid(self.stations_x, y_web, indexing="ij")
        X_rw, Y_rw = np.meshgrid(self.stations_x, y_web, indexing="ij")

        self.components = {
            "top_deck": (X_top, np.full_like(X_top, self.y_top), Z_top),
            "bottom_soffit": (X_bot, np.full_like(X_bot, self.y_bot), Z_bot),
            "left_web": (X_lw, Y_lw, self.z_web_bot_L + (Y_lw - self.y_bot) / (self.y_top - self.y_bot) * (self.z_web_top_L - self.z_web_bot_L)),
            "right_web": (X_rw, Y_rw, self.z_web_bot_R + (Y_rw - self.y_bot) / (self.y_top - self.y_bot) * (self.z_web_top_R - self.z_web_bot_R)),
        }

        self.tendons = []
        tan_theta = (8.28 - 4.86) / (2.0 * 3.85)

        for z_off in [-0.75, 0.0, 0.75]:
            self.tendons.append({
                "name": f"Top Tendon (Z={z_off:+.2f}m)", "family": "Top Deck",
                "y_fn": lambda x: np.full_like(x, 3.65), "z_fn": lambda x, y, zo=z_off: np.full_like(x, zo)
            })

        y_a_list = [3.25, 2.75, 2.25, 1.75]
        y_m_list = [0.60, 0.50, 0.40, 0.30]
        for i, (ya, ym) in enumerate(zip(y_a_list, y_m_list)):
            self.tendons.append({
                "name": f"Left Web Draped {i+1}", "family": "Left Web",
                "y_fn": lambda x, ya=ya, ym=ym: ya - 4.0 * (ya - ym) * (x / self.L) * (1.0 - x / self.L),
                "z_fn": lambda x, y: -(4.86 / 2.0 + y * tan_theta)
            })
            self.tendons.append({
                "name": f"Right Web Draped {i+1}", "family": "Right Web",
                "y_fn": lambda x, ya=ya, ym=ym: ya - 4.0 * (ya - ym) * (x / self.L) * (1.0 - x / self.L),
                "z_fn": lambda x, y: (4.86 / 2.0 + y * tan_theta)
            })

        for z_off in [-2.03, -1.43, 1.43, 2.03]:
            self.tendons.append({
                "name": f"Bottom Tendon (Z={z_off:+.2f}m)", "family": "Bottom Soffit",
                "y_fn": lambda x: np.full_like(x, 0.20), "z_fn": lambda x, y, zo=z_off: np.full_like(x, zo)
            })

        self.bearings = np.array([
            [0.0, self.y_bot, self.z_web_bot_L], [0.0, self.y_bot, self.z_web_bot_R],
            [self.L, self.y_bot, self.z_web_bot_L], [self.L, self.y_bot, self.z_web_bot_R],
        ])

class StatusCard(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)

        self.lbl_title = QtWidgets.QLabel(title.upper())
        self.lbl_title.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8px; letter-spacing:1.2px; background:transparent; border:none;")
        self.lbl_state = QtWidgets.QLabel("—")
        self.lbl_state.setStyleSheet(f"color:{THEME['text']}; font-size:11.5px; font-weight:bold; background:transparent; border:none;")
        self.lbl_metric = QtWidgets.QLabel("")
        self.lbl_metric.setStyleSheet(f"color:{THEME['text_dim']}; font-size:8.5px; font-family:monospace; background:transparent; border:none;")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_state)
        layout.addWidget(self.lbl_metric)
        self._apply_level(0)

    def _apply_level(self, level: int):
        color = LEVEL_COLORS[min(level, 3)]
        self.setStyleSheet(f"QFrame#StatusCard {{ background-color: #ffffff; border-left: 4px solid {color}; border: 1px solid {THEME['border']}; border-radius: 4px; }}")
        self.lbl_state.setStyleSheet(f"color:{color}; font-size:11.5px; font-weight:bold; background:transparent; border:none;")

    def update_state(self, state: str, metric: str, level: int):
        self.lbl_state.setText(state)
        self.lbl_metric.setText(metric)
        self._apply_level(level)

def _section_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text.upper())
    lbl.setStyleSheet(f"color: {THEME['accent']}; font-size: 9px; font-weight: bold; letter-spacing: 1.4px; padding-top: 6px; padding-bottom: 2px; border-top: 1px solid {THEME['border']}; background: transparent;")
    return lbl

def _mono(text: str, color: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-family: Consolas, monospace; background: transparent; border: none;")
    return lbl

class Twin3DViewportCanvas(FigureCanvas):
    def __init__(self, parent=None, width=14, height=6.6, dpi=115):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="#ffffff")
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_3d = self.fig.add_subplot(111, projection="3d", facecolor="#ffffff")
        self.ax_3d.view_init(elev=18, azim=-55)
        self.ax_3d.dist = 5.0
        self.ax_3d.set_box_aspect((59.82, 16.0, 7.0), zoom=1.35)

        self.ax_3d.set_xlim([-1.0, 61.0])
        self.ax_3d.set_ylim([-8.5, 8.5])
        self.ax_3d.set_zlim([-0.6, 4.6])

        self.ax_3d.xaxis.set_major_locator(MultipleLocator(20.0))
        self.ax_3d.yaxis.set_major_locator(MultipleLocator(4.0))
        self.ax_3d.zaxis.set_major_locator(MultipleLocator(2.0))

        self.ax_3d.set_xlabel("Span X (m)", color=THEME["text"], fontsize=9, labelpad=4)
        self.ax_3d.set_ylabel("Transverse Z (m)", color=THEME["text"], fontsize=9, labelpad=4)
        self.ax_3d.set_zlabel("Vertical Y (m)", color=THEME["text"], fontsize=9, labelpad=4)
        self.ax_3d.tick_params(colors=THEME["text"], labelsize=8, pad=1)

        for axis in (self.ax_3d.xaxis, self.ax_3d.yaxis, self.ax_3d.zaxis):
            axis.pane.fill = True
            axis.pane.set_facecolor("#ffffff")
            axis.pane.set_edgecolor(THEME["border"])
        self.ax_3d.grid(True, linestyle=":", color=THEME["grid"], alpha=1.0)
        self.fig.subplots_adjust(left=0.005, right=0.95, top=0.93, bottom=0.01)

    def reset_view(self):
        self.ax_3d.view_init(elev=18, azim=-55)
        self.ax_3d.dist = 5.0
        self.draw_idle()

class TelemetryScopesCanvas(FigureCanvas):
    def __init__(self, parent=None, width=14, height=4.6, dpi=105):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="#ffffff")
        super().__init__(self.fig)
        self.setParent(parent)

        gs = gridspec.GridSpec(2, 3, figure=self.fig, left=0.055, right=0.985, top=0.90, bottom=0.14, hspace=0.55, wspace=0.30)

        def style(ax, title, ylabel, xlabel="Window (s)", color=None):
            ax.set_facecolor("#f8fafc") 
            ax.tick_params(colors=THEME["text"], labelsize=6.5)
            ax.set_title(title, color=color or THEME["accent"], fontsize=8.5, fontweight="bold", pad=4)
            ax.set_ylabel(ylabel, color=THEME["text_dim"], fontsize=7)
            ax.set_xlabel(xlabel, color=THEME["text_dim"], fontsize=7)
            ax.grid(True, linestyle=":", color=THEME["grid"], alpha=1.0)
            for s in ax.spines.values():
                s.set_color(THEME["border"])

        self.ax_acc = self.fig.add_subplot(gs[0, 0])
        style(self.ax_acc, "1. Input Accel (m/s²)", "Accel", color=THEME["info"])
        self.ax_acc.set_ylim(-2.5, 2.5)

        self.ax_strain = self.fig.add_subplot(gs[0, 1])
        style(self.ax_strain, "2. Selected Node Δε₁ (µε)", "Strain", color=THEME["info"])
        self.ax_strain.set_ylim(-200.0, 200.0)
        self.ax_strain.axhline(0.0, color="#64748b", linestyle="-", lw=1.0)
        self.ax_strain.axhline(100.0, color=THEME["warn"], linestyle="--", lw=1.2)
        self.ax_strain.axhline(130.0, color=THEME["crit"], linestyle="-", lw=1.2)

        self.ax_stress = self.fig.add_subplot(gs[0, 2])
        style(self.ax_stress, "3. Selected Node Total σ₁ (MPa)", "Stress", color=THEME["crit"])
        self.ax_stress.set_ylim(-15.0, 8.0)
        self.ax_stress.axhline(0.0, color="#64748b", linestyle="-", lw=1.0)
        self.ax_stress.axhline(4.10, color=THEME["warn"], linestyle="--", lw=1.2)
        self.ax_stress.axhline(5.30, color=THEME["crit"], linestyle="-", lw=1.2)
        self.ax_stress.axhspan(4.10, 5.30, color=THEME["warn"], alpha=0.1)
        self.ax_stress.axhspan(5.30, 8.0, color=THEME["crit"], alpha=0.15)
        self.ax_stress.text(-10.0, 4.4, 'Cracking (4.1)', color=THEME["warn"], fontsize=6.5, fontweight='bold')
        self.ax_stress.text(-10.0, 6.0, 'Rupture (5.3)', color=THEME["crit"], fontsize=6.5, fontweight='bold')

        self.ax_fft = self.fig.add_subplot(gs[1, 0])
        style(self.ax_fft, "4. Vertical FFT Spectrum", "dB", xlabel="Freq (Hz)", color=THEME["violet"])
        self.ax_fft.set_ylim(-80.0, 20.0)

        self.ax_pt = self.fig.add_subplot(gs[1, 1])
        style(self.ax_pt, "5. Tendon ΔN_pt (kN)", "Force", color=THEME["accent_2"])
        self.ax_pt.set_ylim(-150.0, 150.0)

        self.ax_dmg = self.fig.add_subplot(gs[1, 2])
        style(self.ax_dmg, "6. Cumulative Rainflow Damage", "D (log)", color=THEME["violet"])
        self.ax_dmg.set_yscale("log")
        self.ax_dmg.set_ylim(1e-12, 1e-1)

class DigitalTwinGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Structural Digital Twin Dashboard — Multi-Scale Telemetry & Stress Analysis")
        self.resize(1980, 1120)
        self.setStyleSheet(app_qss())

        self.twin = MTHLDigitalTwin()
        try:
            self.active_mask = self.twin.span_mask.squeeze().cpu().numpy().astype(bool)
        except Exception:
            self.active_mask = np.ones(self.twin.coords.shape[0], dtype=bool)

        self.selected_node_id = None
        self.selected_node_cat = None
        self.selected_tendon_elem = None
        self.selected_tendon_xyz = None

        self.sensor_node_id = SENSOR_NODE_ID
        self.node_visibility = {"Deck": True, "Soffit": True, "Web": True, "Tendon": True, "Sensor": True}

        self.bridge_model = RealisticBoxGirderModel(span_length=59.818)
        self.unique_x_stations = self.bridge_model.stations_x

        x_all = self.twin.coords[:, 0]
        self.station_node_indices = [np.where(np.isclose(x_all, ux, atol=0.25))[0] for ux in self.unique_x_stations]

        self._build_tendon_database()

        self.def_scale = 800.0
        self.auto_orbit = False
        self._last_state_version = -1
        self._last_state_time = time.time()
        self._last_ts = 0.0
        self._fps_window = deque(maxlen=30)
        self._shortcuts = []

        self.hist_len = self.twin.window_size
        self.time_hist = np.linspace(-10.24, 0.0, self.hist_len)
        self.d_tendon_hist = deque([1e-12] * self.hist_len, maxlen=self.hist_len)
        self.d_conc_hist = deque([1e-12] * self.hist_len, maxlen=self.hist_len)

        self._build_ui()
        self._init_3d_and_time_plots()
        self._wire_shortcuts()

        self.twin.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_telemetry_ui)
        self.timer.start(33)

    def _build_tendon_database(self):
        x_st = self.bridge_model.stations_x
        nodes_xyz = []
        element_meta = []

        ele_counter = 0
        for t_idx, t in enumerate(self.bridge_model.tendons):
            y_pts = t["y_fn"](x_st)
            z_pts = t["z_fn"](x_st, y_pts)
            for s_idx in range(len(x_st)):
                nodes_xyz.append([x_st[s_idx], y_pts[s_idx], z_pts[s_idx]])
                if s_idx > 0:
                    element_meta.append({
                        "elem_id": ele_counter,
                        "name": f"{t['name']} Seg {s_idx}",
                        "family": t["family"],
                        "mid_xyz": [
                            0.5 * (x_st[s_idx-1] + x_st[s_idx]),
                            0.5 * (y_pts[s_idx-1] + y_pts[s_idx]),
                            0.5 * (z_pts[s_idx-1] + z_pts[s_idx]),
                        ]
                    })
                    ele_counter += 1

        self.tendon_nodes_xyz = np.array(nodes_xyz, dtype=np.float32)  # (300, 3)
        self.tendon_elements = element_meta                         # 285 elements

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._build_header_bar())

        main_row = QtWidgets.QHBoxLayout()
        main_row.setSpacing(10)
        main_row.addWidget(self._build_sidebar())

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)

        self.canvas_3d = Twin3DViewportCanvas(self)
        self.canvas_scopes = TelemetryScopesCanvas(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas_3d)
        splitter.addWidget(self.canvas_scopes)
        splitter.setSizes([640, 380])
        right.addWidget(splitter)

        main_row.addLayout(right, 1)
        root.addLayout(main_row, 1)

    def _build_header_bar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background-color: {THEME['panel']}; border: 2px solid {THEME['border']}; border-radius: 8px;")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(18)

        brand = QtWidgets.QLabel("SHM DIGITAL TWIN ENGINE")
        brand.setStyleSheet(f"color: {THEME['text']}; font-weight: bold; font-size: 12px; background: transparent;")
        h.addWidget(brand)
        h.addWidget(self._vline())

        self.card_hardware = QtWidgets.QLabel(f"DEVICE: {hardware_name}")
        self.card_hardware.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_hardware)
        h.addWidget(self._vline())

        self.card_latency = QtWidgets.QLabel("INFERENCE: -- ms")
        self.card_latency.setStyleSheet(f"color: {THEME['accent_2']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_latency)

        self.card_fps = QtWidgets.QLabel("FPS: --")
        self.card_fps.setStyleSheet(f"color: {THEME['info']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_fps)

        self.card_stale = QtWidgets.QLabel("PACKET: -- ms")
        self.card_stale.setStyleSheet(f"color: {THEME['ok']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_stale)
        h.addWidget(self._vline())

        self.card_selected = QtWidgets.QLabel("SELECTED: none")
        self.card_selected.setStyleSheet(f"color: {THEME['crit']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_selected)
        h.addStretch()

        self.card_sensor = QtWidgets.QLabel(f"INPUT ACCELEROMETER: Node #{SENSOR_NODE_ID:03d}")
        self.card_sensor.setStyleSheet(f"color: {THEME['magenta']}; font-weight: bold; font-size: 10.5px; background: transparent;")
        h.addWidget(self.card_sensor)
        h.addWidget(self._vline())

        self.card_status = QtWidgets.QLabel("SYSTEM ONLINE")
        self.card_status.setStyleSheet(f"color: white; font-weight: bold; font-size: 11px; background: {THEME['ok']}; border-radius: 4px; padding: 4px 10px;")
        h.addWidget(self.card_status)
        self.metric_bar = bar
        return bar

    @staticmethod
    def _vline() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setStyleSheet(f"color: {THEME['border']}; background: {THEME['border']};")
        line.setFixedWidth(1)
        return line

    @staticmethod
    def _hline() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {THEME['border']}; background: {THEME['border']};")
        line.setFixedHeight(1)
        return line

    def _build_sidebar(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setFixedWidth(400)
        scroll.setWidgetResizable(True)

        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        t = QtWidgets.QLabel("STRUCTURAL SURROGATE TWIN")
        t.setStyleSheet(f"font-size: 17px; font-weight: 800; color: {THEME['text']}; background: transparent;")
        v.addWidget(t)
        sub = QtWidgets.QLabel("Full-Field Mechanics · 15 PT Tendons (285 Elements / 300 Nodes)")
        sub.setStyleSheet(f"font-size: 9.5px; color: {THEME['text_dim']}; background: transparent;")
        v.addWidget(sub)
        v.addWidget(self._hline())

        # Sensor alignment info
        v.addWidget(_section_label("Accelerometer Alignment"))
        sensor_box = QtWidgets.QFrame()
        sensor_box.setStyleSheet(f"background: {THEME['panel_2']}; border: 1px solid {THEME['border']}; border-left: 4px solid {THEME['magenta']}; border-radius: 4px;")
        sb = QtWidgets.QVBoxLayout(sensor_box)
        sb.setContentsMargins(10, 6, 10, 6)
        sb.setSpacing(2)
        c_sensor = self.twin.coords[self.sensor_node_id]
        sb.addWidget(_mono(f"Telemetry Mapping: ST-GNN Index #{self.sensor_node_id:03d} (OpenSees #{OPENSEES_SENSOR})", THEME["magenta"]))
        sb.addWidget(_mono(f"X = {c_sensor[0]:6.2f} m   Y = {c_sensor[1]:4.2f} m   Z = {c_sensor[2]:+5.2f} m", THEME["text_dim"]))
        v.addWidget(sensor_box)

        # Failure cards
        v.addWidget(_section_label("Capacity Limit Evaluation"))
        fail_group = QtWidgets.QGroupBox()
        fail_group.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        fl = QtWidgets.QVBoxLayout(fail_group)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(4)
        self.card_mode_a = StatusCard("Flexural Tension — Soffit")
        self.card_mode_b = StatusCard("Deck Compression — Top")
        self.card_mode_c = StatusCard("Web Diagonal Shear")
        self.card_mode_d = StatusCard("Prestress Tendon")
        for c in (self.card_mode_a, self.card_mode_b, self.card_mode_c, self.card_mode_d):
            fl.addWidget(c)
        v.addWidget(fail_group)

        # Inspector
        v.addWidget(_section_label("Entity Inspector (Node / Tendon)"))
        insp_group = QtWidgets.QGroupBox()
        insp_group.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        il = QtWidgets.QVBoxLayout(insp_group)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(3)

        search_layout = QtWidgets.QHBoxLayout()
        self.node_search_input = QtWidgets.QLineEdit()
        self.node_search_input.setPlaceholderText("Enter Node (0-599) or Tendon (E0-E284)")
        self.node_search_input.setStyleSheet(f"border: 1px solid {THEME['border']}; border-radius: 3px; padding: 3px; background: #ffffff;")
        self.btn_search = QtWidgets.QPushButton("Find")
        self.btn_search.setStyleSheet(btn_style("#0ea5e9"))
        self.btn_search.clicked.connect(self._search_entity)
        search_layout.addWidget(self.node_search_input)
        search_layout.addWidget(self.btn_search)
        il.addLayout(search_layout)

        self.lbl_sel_id = QtWidgets.QLabel("Entity: —")
        self.lbl_sel_id.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold; font-size: 11px; background: transparent;")
        self.lbl_sel_cat = QtWidgets.QLabel("Type: —")
        self.lbl_sel_cat.setStyleSheet(f"color: {THEME['text']}; font-weight: bold; font-size: 9.5px; background: transparent;")
        il.addWidget(self.lbl_sel_id)
        il.addWidget(self.lbl_sel_cat)

        resp_frame = QtWidgets.QFrame()
        resp_frame.setStyleSheet(f"background: {THEME['panel_2']}; border: 1px solid {THEME['border']}; border-radius: 4px;")
        rg = QtWidgets.QGridLayout(resp_frame)
        rg.setContentsMargins(10, 6, 10, 6)
        rg.setHorizontalSpacing(8)
        rg.setVerticalSpacing(1)

        def _rlbl(color):
            lbl = QtWidgets.QLabel("—")
            lbl.setStyleSheet(f"color: {color}; font-size: 9.5px; font-family: Consolas, monospace; background: transparent;")
            return lbl

        self.lbl_sel_s1 = _rlbl(THEME["crit"])
        self.lbl_sel_s3 = _rlbl(THEME["info"])
        self.lbl_sel_vm = _rlbl(THEME["accent_2"])
        self.lbl_sel_e1 = _rlbl(THEME["crit"])
        self.lbl_sel_e3 = _rlbl(THEME["info"])
        self.lbl_sel_uy = _rlbl(THEME["ok"])

        rg.addWidget(self.lbl_sel_s1, 0, 0)
        rg.addWidget(self.lbl_sel_s3, 1, 0)
        rg.addWidget(self.lbl_sel_vm, 2, 0)
        rg.addWidget(self.lbl_sel_e1, 0, 1)
        rg.addWidget(self.lbl_sel_e3, 1, 1)
        rg.addWidget(self.lbl_sel_uy, 2, 1)
        il.addWidget(resp_frame)

        self.btn_clear_sel = QtWidgets.QPushButton("Clear Selection [Esc]")
        self.btn_clear_sel.setStyleSheet(btn_style("#475569"))
        self.btn_clear_sel.clicked.connect(self._clear_selection)
        il.addWidget(self.btn_clear_sel)
        v.addWidget(insp_group)

        # Filters
        v.addWidget(_section_label("Domain Visibility Filters"))
        filter_group = QtWidgets.QGroupBox()
        filter_group.setStyleSheet("QGroupBox { border: none; padding: 0; margin: 0; }")
        flt = QtWidgets.QVBoxLayout(filter_group)
        flt.setContentsMargins(0, 0, 0, 0)
        flt.setSpacing(3)

        self.chk_cat = {}
        cat_colors_ui = {"Deck": "#1d4ed8", "Soffit": "#15803d", "Web": "#c2410c", "Tendon": "#be185d", "Sensor": "#a21caf"}
        cat_counts = {
            "Deck": len(self.twin.idx_by_cat["Deck"]), "Soffit": len(self.twin.idx_by_cat["Soffit"]),
            "Web": len(self.twin.idx_by_cat["Web"]), "Tendon": len(self.tendon_elements), "Sensor": 1
        }
        for cat in ["Deck", "Soffit", "Web", "Tendon", "Sensor"]:
            chk = QtWidgets.QCheckBox(f"{cat} ({cat_counts[cat]} items)")
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {cat_colors_ui[cat]}; font-weight: bold; font-size: 9px;")
            chk.toggled.connect(lambda val, c=cat: self._toggle_category(c, val))
            flt.addWidget(chk)
            self.chk_cat[cat] = chk
        v.addWidget(filter_group)

        # Fatigue Panel
        v.addWidget(_section_label("Fatigue Life Tracking (Rainflow)"))
        fat_frame = QtWidgets.QFrame()
        fat_frame.setStyleSheet(f"background: {THEME['panel_2']}; border: 1px solid {THEME['border']}; border-radius: 4px;")
        fa = QtWidgets.QVBoxLayout(fat_frame)
        fa.setContentsMargins(10, 6, 10, 6)
        fa.setSpacing(2)

        self.lbl_fatigue_p = QtWidgets.QLabel("PT Tendon   D = 0.000e+00")
        self.lbl_fatigue_p.setStyleSheet(f"color: {THEME['violet']}; font-weight: bold; font-size: 9px; font-family: Consolas, monospace; background: transparent;")
        self.lbl_fatigue_c = QtWidgets.QLabel("Concrete    D = 0.000e+00")
        self.lbl_fatigue_c.setStyleSheet(f"color: {THEME['accent_2']}; font-weight: bold; font-size: 9px; font-family: Consolas, monospace; background: transparent;")
        self.lbl_cycles = QtWidgets.QLabel("Closed cycles = 0")
        self.lbl_cycles.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 8.5px; font-family: Consolas, monospace; background: transparent;")
        
        fa.addWidget(self.lbl_fatigue_p)
        fa.addWidget(self.lbl_fatigue_c)
        fa.addWidget(self.lbl_cycles)
        v.addWidget(fat_frame)

        # Stream Controls
        v.addWidget(_section_label("Telemetry Stream Control"))
        tel_frame = QtWidgets.QFrame()
        tel_frame.setStyleSheet(f"background: {THEME['panel_2']}; border: 1px solid {THEME['border']}; border-radius: 4px;")
        tl = QtWidgets.QVBoxLayout(tel_frame)
        tl.setContentsMargins(10, 6, 10, 6)
        tl.setSpacing(4)

        self.lbl_accel = QtWidgets.QLabel("ax: —   ay: —   az: —")
        self.lbl_accel.setStyleSheet(f"color: {THEME['info']}; font-family: Consolas, monospace; font-size: 9px; background: transparent;")
        tl.addWidget(self.lbl_accel)

        self.lbl_time_pos = QtWidgets.QLabel("Record: 0.00 / 86400.00 s")
        self.lbl_time_pos.setStyleSheet(f"color: {THEME['violet']}; font-family: Consolas, monospace; font-size: 9px; background: transparent;")
        tl.addWidget(self.lbl_time_pos)

        speed_row = QtWidgets.QHBoxLayout()
        speed_lbl = QtWidgets.QLabel("Speed")
        speed_lbl.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 9.5px; background: transparent;")
        self.combo_speed = QtWidgets.QComboBox()
        self.combo_speed.addItems(["1× (100 Hz)", "2×", "5×", "10×"])
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        speed_row.addWidget(speed_lbl)
        speed_row.addWidget(self.combo_speed, 1)
        tl.addLayout(speed_row)

        self.btn_toggle_stream = QtWidgets.QPushButton("Pause Stream [Space]")
        self.btn_toggle_stream.setStyleSheet(btn_style("#0284c7"))
        self.btn_toggle_stream.clicked.connect(self._toggle_stream)
        tl.addWidget(self.btn_toggle_stream)
        v.addWidget(tel_frame)

        # Camera & Scaling
        v.addWidget(_section_label("Visual Scaling & Camera"))
        vis_frame = QtWidgets.QFrame()
        vis_frame.setStyleSheet(f"background: {THEME['panel_2']}; border: 1px solid {THEME['border']}; border-radius: 4px;")
        vl = QtWidgets.QVBoxLayout(vis_frame)
        vl.setContentsMargins(10, 6, 10, 6)
        vl.setSpacing(4)

        scale_row = QtWidgets.QHBoxLayout()
        scale_lbl = QtWidgets.QLabel("Sag ×")
        scale_lbl.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 9.5px; background: transparent;")
        self.spin_scale = QtWidgets.QSpinBox()
        self.spin_scale.setRange(50, 3000)
        self.spin_scale.setSingleStep(50)
        self.spin_scale.setValue(int(self.def_scale))
        self.spin_scale.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(scale_lbl)
        scale_row.addWidget(self.spin_scale, 1)
        vl.addLayout(scale_row)

        cam_row = QtWidgets.QHBoxLayout()
        self.btn_reset_view = QtWidgets.QPushButton("Reset View [V]")
        self.btn_reset_view.setStyleSheet(btn_style("#0891b2"))
        self.btn_reset_view.clicked.connect(self._reset_camera)
        cam_row.addWidget(self.btn_reset_view)
        self.btn_toggle_orbit = QtWidgets.QPushButton("Orbit: OFF")
        self.btn_toggle_orbit.setStyleSheet(btn_style("#64748b"))
        self.btn_toggle_orbit.clicked.connect(self._toggle_orbit)
        cam_row.addWidget(self.btn_toggle_orbit)
        vl.addLayout(cam_row)
        v.addWidget(vis_frame)

        v.addStretch()
        scroll.setWidget(content)
        return scroll

    def _init_3d_and_time_plots(self):
        # 1. Wireframe mesh
        self.poly_mesh = Poly3DCollection(
            [], facecolors="none", edgecolors=MESH_EDGE,
            linewidths=0.25, alpha=MESH_ALPHA, antialiased=True,
        )
        self.canvas_3d.ax_3d.add_collection3d(self.poly_mesh)

        # 2. 15 Continuous Tendon Lines
        x_dense = np.linspace(0.0, self.bridge_model.L, 100)
        self.tendon_lines = []
        for t in self.bridge_model.tendons:
            y_d = t["y_fn"](x_dense)
            z_d = t["z_fn"](x_dense, y_d)
            (line,) = self.canvas_3d.ax_3d.plot(
                x_dense, z_d, y_d, color="#be185d", lw=1.6, alpha=0.85,
                label="PT Tendons" if len(self.tendon_lines) == 0 else ""
            )
            self.tendon_lines.append(line)

        # 3. 300 Discrete Tendon Nodes
        t_nodes = self.tendon_nodes_xyz
        self.tendon_nodes_scatter = self.canvas_3d.ax_3d.scatter(
            t_nodes[:, 0], t_nodes[:, 2], t_nodes[:, 1],
            color="#1d4ed8", s=8, alpha=0.9, zorder=6, depthshade=False,
            label="PT Nodes (300)"
        )

        # 4. Bearings
        b_pts = self.bridge_model.bearings
        self.canvas_3d.ax_3d.scatter(
            b_pts[:, 0], b_pts[:, 2], b_pts[:, 1],
            marker="^", s=140, color="#15803d", edgecolor="#052e16", lw=1.2, depthshade=False, label="Bearings"
        )

        # 5. Bridge Nodes
        self.node_scatters = {}
        self.node_cat_colors = {"Deck": "#1d4ed8", "Soffit": "#15803d", "Web": "#c2410c"}
        for cat, color in self.node_cat_colors.items():
            idx = self.twin.idx_by_cat[cat]
            if len(idx) == 0: continue
            xyz = self.twin.coords[idx]
            sc = self.canvas_3d.ax_3d.scatter(
                xyz[:, 0], xyz[:, 2], xyz[:, 1], facecolors=color, edgecolors="white",
                s=44, alpha=0.88, linewidths=0.85, depthshade=False, picker=10, pickradius=14, label=f"{cat} ({len(idx)})"
            )
            sc._category = cat
            sc._node_ids = idx
            self.node_scatters[cat] = sc

        # 6. Discrete Tendon Centroids
        elem_mids = np.array([e["mid_xyz"] for e in self.tendon_elements], dtype=np.float32)
        self.tendon_elem_scatter = self.canvas_3d.ax_3d.scatter(
            elem_mids[:, 0], elem_mids[:, 2], elem_mids[:, 1],
            color="#be185d", s=28, alpha=0.0, depthshade=False, picker=10, pickradius=12
        )
        self.tendon_elem_scatter._category = "Tendon"

        # 7. Accelerometer Mount Marker (Node #309 / ST-GNN #278)
        c_sensor = self.twin.coords[self.sensor_node_id]
        self.sensor_marker = self.canvas_3d.ax_3d.scatter(
            [c_sensor[0]], [c_sensor[2]], [c_sensor[1]],
            s=650, marker="*", color="#a21caf", edgecolors="#0f172a", linewidths=1.8, depthshade=False, zorder=95,
            label=f"Sensor #{OPENSEES_SENSOR}"
        )
        self.sensor_halo = self.canvas_3d.ax_3d.scatter(
            [c_sensor[0]], [c_sensor[2]], [c_sensor[1]],
            s=1150, marker="o", facecolors="none", edgecolors="#a21caf", linewidths=1.6, alpha=0.5, depthshade=False, zorder=94
        )
        self.sensor_cross_x, = self.canvas_3d.ax_3d.plot(
            [c_sensor[0], c_sensor[0]], [c_sensor[2] - SENSOR_ARM, c_sensor[2] + SENSOR_ARM], [c_sensor[1], c_sensor[1]],
            color="#a21caf", lw=1.2, alpha=0.9, zorder=94
        )
        self.sensor_cross_y, = self.canvas_3d.ax_3d.plot(
            [c_sensor[0], c_sensor[0]], [c_sensor[2], c_sensor[2]], [c_sensor[1] - 0.6, c_sensor[1] + 0.6],
            color="#a21caf", lw=1.2, alpha=0.9, zorder=94
        )
        self.sensor_drop_line, = self.canvas_3d.ax_3d.plot(
            [c_sensor[0], c_sensor[0]], [c_sensor[2], c_sensor[2]], [c_sensor[1], -0.6],
            color="#a21caf", lw=1.0, ls=":", alpha=0.6
        )
        self.sensor_annot = self.canvas_3d.ax_3d.annotate(
            f"SENSOR #{OPENSEES_SENSOR}\n(ST-GNN #{self.sensor_node_id:03d})",
            xy=(0.5, 0.5), xytext=(0, 20), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="#a21caf", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.32", fc="#ffffff", ec="#a21caf", lw=1.1, alpha=0.96), zorder=200
        )

        self.selected_marker = self.canvas_3d.ax_3d.scatter(
            [np.nan], [np.nan], [np.nan], s=340, facecolors="none", edgecolors="#dc2626", linewidths=2.8, marker="o", zorder=100
        )
        self.selected_marker.set_visible(False)

        leg = self.canvas_3d.ax_3d.legend(
            loc="upper right", bbox_to_anchor=(0.95, 1.00), framealpha=1.0,
            edgecolor=THEME["border"], fontsize=7.0, labelcolor=THEME["text"], facecolor="#ffffff",
            borderpad=0.55, handletextpad=0.55, labelspacing=0.42
        )
        plt.setp(leg.get_texts(), fontsize=7)

        self.canvas_3d.ax_3d.set_title(
            "Multi-Span Infrastructure · 15 PT Tendons (285 Elements / 300 Nodes)",
            fontsize=11.5, fontweight="bold", pad=8, color=THEME["text"], loc="left"
        )

        self.caption_badge = self.canvas_3d.ax_3d.text2D(
            0.02, 0.03, "", transform=self.canvas_3d.ax_3d.transAxes,
            fontsize=9, fontweight="bold", color=THEME["text"], va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=THEME["border"], lw=0.9, alpha=0.95), zorder=200
        )

        # Scopes setup
        th = self.time_hist
        (self.line_ay,) = self.canvas_scopes.ax_acc.plot(th, np.zeros(self.hist_len), color=THEME["ok"], lw=1.2, label="ay")
        (self.line_ax,) = self.canvas_scopes.ax_acc.plot(th, np.zeros(self.hist_len), color=THEME["info"], lw=0.8, alpha=0.85, label="ax")
        (self.line_az,) = self.canvas_scopes.ax_acc.plot(th, np.zeros(self.hist_len), color=THEME["magenta"], lw=0.8, alpha=0.85, label="az")
        self.canvas_scopes.ax_acc.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=3, fontsize=6.0, framealpha=0.9, labelcolor=THEME["text"], facecolor="#ffffff", edgecolor=THEME["border"], borderpad=0.28, handlelength=1.4, columnspacing=0.9)

        (self.line_strain,) = self.canvas_scopes.ax_strain.plot(th, np.zeros(self.hist_len), color=THEME["info"], lw=1.5)
        (self.line_stress,) = self.canvas_scopes.ax_stress.plot(th, np.zeros(self.hist_len), color=THEME["crit"], lw=1.5)
        (self.line_fft,) = self.canvas_scopes.ax_fft.plot(self.twin.fft_freqs, np.full_like(self.twin.fft_freqs, -80.0), color=THEME["violet"], lw=1.2)
        self.canvas_scopes.ax_fft.set_xlim(0, min(50.0, self.twin.fft_freqs.max()))
        (self.line_pt,) = self.canvas_scopes.ax_pt.plot(th, np.zeros(self.hist_len), color=THEME["accent_2"], lw=1.5)

        self.line_d_t = self.canvas_scopes.ax_dmg.fill_between(self.time_hist, 1e-12, [1e-12]*self.hist_len, color=THEME["violet"], alpha=0.3, label="PT")
        self.line_d_c = self.canvas_scopes.ax_dmg.fill_between(self.time_hist, 1e-12, [1e-12]*self.hist_len, color=THEME["accent_2"], alpha=0.3, label="Concrete")
        self.canvas_scopes.ax_dmg.legend(loc="upper left", fontsize=6.0, framealpha=0.9, labelcolor=THEME["text"], facecolor="#ffffff", edgecolor=THEME["border"], borderpad=0.28, handlelength=1.4)

        self.canvas_3d.draw()
        self.canvas_scopes.draw()

        self.canvas_3d.mpl_connect("pick_event", self._on_3d_pick)
        self.canvas_3d.mpl_connect("button_press_event", self._on_3d_click)

    def _wire_shortcuts(self):
        QShortcut = getattr(QtGui, "QShortcut", None)
        if QShortcut is None: QShortcut = QtWidgets.QShortcut
        def bind(key, fn):
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)
        bind("Space", self._toggle_stream)
        bind("R", self._reset_scale)
        bind("S", self._snapshot)
        bind("A", self._toggle_orbit)
        bind("V", self._reset_camera)
        bind("F", self._toggle_fullscreen)
        bind("Esc", self._clear_selection)

    def _reset_scale(self): self.spin_scale.setValue(800)
    def _toggle_fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def _reset_camera(self): self.canvas_3d.reset_view(); self.card_status.setText("VIEW RESET")
    def _toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit
        if self.auto_orbit: self.btn_toggle_orbit.setText("Orbit: ON"); self.btn_toggle_orbit.setStyleSheet(btn_style("#0891b2"))
        else: self.btn_toggle_orbit.setText("Orbit: OFF"); self.btn_toggle_orbit.setStyleSheet(btn_style("#64748b"))

    def _search_entity(self):
        txt = self.node_search_input.text().strip()
        try:
            if txt.upper().startswith("E"):
                elem_id = int(txt[1:])
                if 0 <= elem_id < len(self.tendon_elements):
                    self._select_tendon(elem_id, self.tendon_elements[elem_id]["mid_xyz"])
            else:
                nid = int(txt)
                if 0 <= nid < self.twin.coords.shape[0]:
                    self._select_node(nid)
        except ValueError:
            pass

    def _toggle_category(self, cat: str, visible: bool):
        self.node_visibility[cat] = visible
        if cat == "Tendon":
            for line in self.tendon_lines: line.set_visible(visible)
            self.tendon_nodes_scatter.set_visible(visible)
        elif cat == "Sensor":
            for art in (self.sensor_marker, self.sensor_halo, self.sensor_cross_x, self.sensor_cross_y, self.sensor_drop_line, self.sensor_annot):
                art.set_visible(visible)
        elif cat in self.node_scatters:
            self.node_scatters[cat].set_visible(visible)

    def _clear_selection(self):
        self.selected_node_id = None
        self.selected_node_cat = None
        self.selected_tendon_elem = None
        self.selected_tendon_xyz = None
        self.selected_marker.set_visible(False)
        self.lbl_sel_id.setText("Entity: —")
        self.lbl_sel_cat.setText("Type: —")
        for lbl in (self.lbl_sel_s1, self.lbl_sel_s3, self.lbl_sel_vm, self.lbl_sel_e1, self.lbl_sel_e3, self.lbl_sel_uy): 
            lbl.setText("—")
        self.card_selected.setText("SELECTED: none")
        self.card_status.setText("SYSTEM ONLINE")
        self.card_status.setStyleSheet(f"color: white; font-weight: bold; font-size: 11px; background: {THEME['ok']}; border-radius: 4px; padding: 4px 10px;")

    def _select_node(self, nid: int):
        self.selected_node_id = int(nid)
        self.selected_tendon_elem = None
        self.selected_tendon_xyz = None
        cat = str(self.twin.node_category[nid])
        self.selected_node_cat = cat
        
        tag = f" · SENSOR #{OPENSEES_SENSOR}" if nid == self.sensor_node_id else ""
        self.lbl_sel_id.setText(f"Bridge Node: #{nid:03d}{tag}")
        self.lbl_sel_cat.setText(f"Type: {cat} Shell Node")
        self.selected_marker.set_visible(True)
        self.card_selected.setText(f"SELECTED: Node #{nid:03d} ({cat})")
        self.card_status.setText(f"LOCKED · Node #{nid:03d}")
        self.card_status.setStyleSheet(f"color: white; font-weight: bold; font-size: 11px; background: {THEME['accent']}; border-radius: 4px; padding: 4px 10px;")

    def _select_tendon(self, elem_idx: int, xyz):
        self.selected_node_id = None
        self.selected_node_cat = "Tendon"
        self.selected_tendon_elem = int(elem_idx)
        self.selected_tendon_xyz = xyz
        
        elem_data = self.tendon_elements[elem_idx]
        self.lbl_sel_id.setText(f"Tendon Element: E{elem_idx:03d}")
        self.lbl_sel_cat.setText(f"{elem_data['name']} ({elem_data['family']})")
        self.selected_marker.set_visible(True)
        self.card_selected.setText(f"SELECTED: Tendon E{elem_idx:03d}")
        self.card_status.setText(f"LOCKED · Tendon E{elem_idx:03d}")
        self.card_status.setStyleSheet(f"color: white; font-weight: bold; font-size: 11px; background: {THEME['accent']}; border-radius: 4px; padding: 4px 10px;")

    def _on_3d_click(self, event):
        if event.button == 1 and self.auto_orbit: self._toggle_orbit()

    def _on_3d_pick(self, event):
        artist = event.artist
        try:
            cat = getattr(artist, "_category", None)
            if cat is None: return
            ind = int(event.ind[0]) if hasattr(event, "ind") and len(event.ind) > 0 else 0
            if cat == "Tendon": 
                self._select_tendon(ind, self.tendon_elements[ind]["mid_xyz"])
            else:
                ids = artist._node_ids
                if ind < len(ids): self._select_node(int(ids[ind]))
        except Exception as e: 
            print(f"[pick] ignored: {e}")

    def _on_speed_changed(self, idx): self.twin.playback_speed = [1.0, 2.0, 5.0, 10.0][idx]
    def _on_scale_changed(self, val): self.def_scale = float(val)

    def _toggle_stream(self):
        if self.twin.running:
            self.twin.stop()
            self.btn_toggle_stream.setText("Resume Stream [Space]")
            self.btn_toggle_stream.setStyleSheet(btn_style(THEME["ok"]))
        else:
            self.twin.start()
            self.btn_toggle_stream.setText("Pause Stream [Space]")
            self.btn_toggle_stream.setStyleSheet(btn_style("#0284c7"))

    def _snapshot(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("snapshots", exist_ok=True)
        try:
            self.canvas_3d.fig.savefig(f"snapshots/twin_3d_{ts}.png", dpi=160, facecolor="#ffffff")
            self.canvas_scopes.fig.savefig(f"snapshots/twin_scopes_{ts}.png", dpi=160, facecolor="#ffffff")
            self.card_status.setText(f"SNAPSHOT {ts}")
        except Exception as e: 
            self.card_status.setText(f"SNAPSHOT FAILED: {e}")

    def _update_telemetry_ui(self):
        now = time.perf_counter()
        self._fps_window.append(now)
        if len(self._fps_window) >= 2:
            self.card_fps.setText(f"FPS: {(len(self._fps_window) - 1) / (self._fps_window[-1] - self._fps_window[0]):4.1f}")

        with self.twin.lock: 
            st = dict(self.twin.live_state)

        if st["timestamp"] != self._last_ts: 
            self._last_ts = st["timestamp"]
            self._last_state_time = time.time()
        stale_ms = (time.time() - self._last_state_time) * 1000.0
        self.card_stale.setText(f"PACKET: {stale_ms:4.0f} ms")
        self.card_stale.setStyleSheet(f"color: {THEME['ok'] if stale_ms < 300 else (THEME['warn'] if stale_ms < 1000 else THEME['crit'])}; font-weight: bold; font-size: 10.5px; background: transparent;")

        acc_win = st["accel_window"]
        self.line_ax.set_ydata(acc_win[0])
        self.line_ay.set_ydata(acc_win[1])
        self.line_az.set_ydata(acc_win[2])

        disp_traj = st["disp_trajectory"]
        cur_sag_mm = float(st.get("delta_uy_mm", np.zeros(1))[-1])
        uy_fem_mm = disp_traj[:, 1, -1] * 1000.0
        station_sag_mm = np.array([np.mean(uy_fem_mm[idx]) if len(idx) > 0 else 0.0 for idx in self.station_node_indices])
        if np.ptp(station_sag_mm) < 1e-4 and abs(cur_sag_mm) > 1e-4: 
            station_sag_mm = cur_sag_mm * np.sin(np.pi * self.unique_x_stations / self.bridge_model.L)

        all_quads = []
        for X, Y, Z in self.bridge_model.components.values():
            Y_def = Y + (np.interp(X, self.unique_x_stations, station_sag_mm) / 1000.0) * self.def_scale
            all_quads.append(np.stack([
                np.stack([X[:-1, :-1], Z[:-1, :-1], Y_def[:-1, :-1]], axis=-1),
                np.stack([X[1:, :-1], Z[1:, :-1], Y_def[1:, :-1]], axis=-1),
                np.stack([X[1:, 1:], Z[1:, 1:], Y_def[1:, 1:]], axis=-1),
                np.stack([X[:-1, 1:], Z[:-1, 1:], Y_def[:-1, 1:]], axis=-1)
            ], axis=-2).reshape(-1, 4, 3))

        verts = np.concatenate(all_quads, axis=0)
        self.poly_mesh.set_verts(verts)
        self.poly_mesh.set_facecolor(np.zeros((len(verts), 4)))
        self.poly_mesh.set_edgecolor(MESH_EDGE)

        # Deform 15 Continuous Tendon Lines
        x_dense = np.linspace(0.0, self.bridge_model.L, 100)
        for idx, t in enumerate(self.bridge_model.tendons):
            y_base = t["y_fn"](x_dense)
            z_base = t["z_fn"](x_dense, y_base)
            y_def = y_base + (np.interp(x_dense, self.unique_x_stations, station_sag_mm) / 1000.0) * self.def_scale
            self.tendon_lines[idx].set_data(x_dense, z_base)
            self.tendon_lines[idx].set_3d_properties(y_def)

        # Deform 300 Discrete Tendon Nodes
        t_sag = np.interp(self.tendon_nodes_xyz[:, 0], self.unique_x_stations, station_sag_mm) / 1000.0
        t_y_def = self.tendon_nodes_xyz[:, 1] + t_sag * self.def_scale
        self.tendon_nodes_scatter._offsets3d = (
            self.tendon_nodes_xyz[:, 0], 
            self.tendon_nodes_xyz[:, 2], 
            t_y_def
        )

        # Accelerometer Crosshairs
        if self.node_visibility.get("Sensor", True):
            c_s = self.twin.coords[self.sensor_node_id]
            sy = c_s[1] + disp_traj[self.sensor_node_id, 1, -1] * self.def_scale
            self.sensor_marker._offsets3d = ([c_s[0]], [c_s[2]], [sy])
            self.sensor_halo._offsets3d = ([c_s[0]], [c_s[2]], [sy])
            self.sensor_cross_x.set_data([c_s[0], c_s[0]], [c_s[2] - SENSOR_ARM, c_s[2] + SENSOR_ARM])
            self.sensor_cross_x.set_3d_properties([sy, sy])
            self.sensor_cross_y.set_data([c_s[0], c_s[0]], [c_s[2], c_s[2]])
            self.sensor_cross_y.set_3d_properties([sy - 0.6, sy + 0.6])
            self.sensor_drop_line.set_data([c_s[0], c_s[0]], [c_s[2], c_s[2]])
            self.sensor_drop_line.set_3d_properties([sy, -0.6])
            try:
                x2, y2, _ = proj3d.proj_transform(c_s[0], c_s[2], sy, self.canvas_3d.ax_3d.get_proj())
                self.sensor_annot.xy = (x2, y2)
            except Exception: pass

        # Structural limit indicators
        fe = st.get("failure_eval", {})
        if fe and hasattr(self, "card_mode_a"):
            self.card_mode_a.update_state(fe.get("mode_a_state", "—"), f"{fe.get('total_s1_soffit', 0.0):+.2f} MPa", fe.get("mode_a_level", 0))
            self.card_mode_b.update_state(fe.get("mode_b_state", "—"), f"{fe.get('total_s3_deck', 0.0):+.2f} MPa", fe.get("mode_b_level", 0))
            self.card_mode_c.update_state(fe.get("mode_c_state", "—"), f"{fe.get('total_s1_web', 0.0):+.2f} MPa", fe.get("mode_c_level", 0))
            self.card_mode_d.update_state(fe.get("mode_d_state", "—"), f"{fe.get('total_sigma_p', 0.0):.1f} MPa", fe.get("mode_d_level", 0))
            self.lbl_fatigue_p.setText(f"PT Tendon   D = {fe.get('D_tendon', 0.0):.3e}")
            self.lbl_fatigue_c.setText(f"Concrete    D = {fe.get('D_conc_comp', 0.0):.3e}")
            self.lbl_cycles.setText(f"Closed cycles = {fe.get('total_cycles', 0)}")
            self.metric_bar.setStyleSheet(f"background-color: {THEME['panel']}; border: 2px solid {LEVEL_COLORS[min(max(fe.get('mode_a_level', 0), fe.get('mode_b_level', 0), fe.get('mode_c_level', 0), fe.get('mode_d_level', 0)), 3)]}; border-radius: 8px;")

        # Active Node Scopes
        target_nid = self.selected_node_id if self.selected_node_id is not None else self.twin.soffit_node
        target_cat = self.twin.node_category[target_nid]
        
        e1_hist = st.get("e1_trajectory", np.zeros((self.twin.coords.shape[0], self.hist_len)))[target_nid, :]
        s1_hist = st.get("s1_trajectory", np.zeros((self.twin.coords.shape[0], self.hist_len)))[target_nid, :]
        base_s1 = -10.5 if target_cat == "Soffit" else (-7.2 if target_cat == "Deck" else -5.0)
        s1_total_hist = base_s1 + s1_hist
        
        self.canvas_scopes.ax_strain.set_title(f"2. Node #{target_nid:03d} ({target_cat}) Δε₁ (µε)", color=THEME["info"], fontsize=9, fontweight="bold", pad=6)
        self.canvas_scopes.ax_stress.set_title(f"3. Node #{target_nid:03d} ({target_cat}) Total σ₁ (MPa)", color=THEME["crit"], fontsize=9, fontweight="bold", pad=6)
        self.line_strain.set_ydata(e1_hist)
        self.line_stress.set_ydata(s1_total_hist)
        self.line_fft.set_ydata(st.get("fft_mag_db", np.full_like(self.twin.fft_freqs, -80.0)))
        
        target_e = self.selected_tendon_elem if self.selected_tendon_elem is not None else 18
        self.canvas_scopes.ax_pt.set_title(f"5. Tendon E{target_e:03d} ΔN_pt (kN)", color=THEME["accent_2"], fontsize=8.5, fontweight="bold", pad=4)
        pt_hist = st.get("pt_trajectory", np.zeros((self.twin.n_tendon, self.hist_len)))[target_e, :]
        self.line_pt.set_ydata(pt_hist)

        for coll in list(self.canvas_scopes.ax_dmg.collections): 
            coll.remove()
        self.d_tendon_hist.append(max(1e-12, float(fe.get("D_tendon", 1e-12))))
        self.d_conc_hist.append(max(1e-12, float(fe.get("D_conc_comp", 1e-12))))
        self.canvas_scopes.ax_dmg.fill_between(self.time_hist, 1e-12, list(self.d_tendon_hist), color=THEME["violet"], alpha=0.3)
        self.canvas_scopes.ax_dmg.fill_between(self.time_hist, 1e-12, list(self.d_conc_hist), color=THEME["accent_2"], alpha=0.3)

        # Inspector updates
        if self.selected_node_id is not None:
            nid = self.selected_node_id
            s1 = float(st.get("s1_trajectory", np.zeros((self.twin.coords.shape[0], 1)))[nid, -1])
            s3 = float(st.get("s3_trajectory", np.zeros((self.twin.coords.shape[0], 1)))[nid, -1])
            vm = float(st.get("vm_trajectory", np.zeros((self.twin.coords.shape[0], 1)))[nid, -1])
            e1 = float(st.get("e1_trajectory", np.zeros((self.twin.coords.shape[0], 1)))[nid, -1])
            e3 = float(st.get("e3_trajectory", np.zeros((self.twin.coords.shape[0], 1)))[nid, -1])
            uy = float(disp_traj[nid, 1, -1]) * 1e3
            c = self.twin.coords[nid]
            self.selected_marker._offsets3d = ([c[0]], [c[2]], [c[1] + disp_traj[nid, 1, -1] * self.def_scale])

            self.lbl_sel_s1.setText(f"  Total σ₁ {base_s1+s1:+6.2f} / 4.10 MPa (Crack)")
            self.lbl_sel_s3.setText(f"  Total σ₃ {base_s1+s3:+6.2f} / -40.0 MPa (Crush)")
            self.lbl_sel_vm.setText(f"  Δσ_vM    {vm:+9.3f} MPa")
            self.lbl_sel_e1.setText(f"  Δε₁      {e1:+9.1f} µε")
            self.lbl_sel_e3.setText(f"  Δε₃      {e3:+9.1f} µε")
            self.lbl_sel_uy.setText(f"  Δu_y     {uy:+9.3f} mm")

        elif self.selected_tendon_elem is not None:
            e = self.selected_tendon_elem
            tf = float(st.get("tendon_force_kn", np.zeros(self.twin.n_tendon))[e])
            ts = float(st.get("tendon_stress_mpa", np.zeros(self.twin.n_tendon))[e])
            tst = self.twin.post_processor.f_pe + ts

            self.lbl_sel_s1.setText(f"  ΔN_pt {tf:+9.3f} kN")
            self.lbl_sel_s3.setText(f"  Δσ_p  {ts:+9.3f} MPa")
            self.lbl_sel_vm.setText(f"  Total {tst:+6.1f} / 1581 MPa (Yield)")
            self.lbl_sel_e1.setText(f"  f_pu  {1860:9.0f} MPa")
            self.lbl_sel_e3.setText(f"  f_py  {1581:9.0f} MPa")
            self.lbl_sel_uy.setText("")

            xyz = self.selected_tendon_xyz
            s_sag = np.interp(xyz[0], self.unique_x_stations, station_sag_mm) / 1000.0
            self.selected_marker._offsets3d = ([xyz[0]], [xyz[2]], [xyz[1] + s_sag * self.def_scale])

        t_curr = st.get("telemetry_time_s", 0.0) % 10.24
        peak_sag = float(np.min(uy_fem_mm)) if len(uy_fem_mm) else 0.0
        self.caption_badge.set_text(
            f"15 PT Tendons (285 Ele / 300 Nodes) · Sag ×{self.def_scale:.0f}"
            f" · t {t_curr:5.2f} s · Peak {abs(peak_sag):5.2f} mm"
        )

        acc = st.get("current_accel", np.zeros(3))
        self.lbl_accel.setText(f"ax {acc[0]:+6.3f}   ay {acc[1]:+6.3f}   az {acc[2]:+6.3f}  m/s²")

        if self.auto_orbit: self.canvas_3d.ax_3d.azim = (self.canvas_3d.ax_3d.azim + 0.15) % 360
        if st["state_version"] != self._last_state_version:
            self._last_state_version = st["state_version"]
            self.canvas_scopes.draw_idle()
        self.canvas_3d.draw_idle()

    def closeEvent(self, event):
        self.twin.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = DigitalTwinGUI()
    gui.show()
    sys.exit(app.exec())