"""
STRUCTURAL HEALTH MONITORING: MASTER THEME & VISUAL TOKENS
File: mthl_theme.py
Theme: High-Contrast Light Engineering Palette
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

THEME = {
    "bg":       "#f1f5f9",   # Slate-100 window background
    "panel":    "#ffffff",   # Pure white panel container
    "panel_2":  "#f8fafc",   # Slate-50 scope background
    "border":   "#cbd5e1",   # Slate-300 border lines
    "grid":     "#e2e8f0",   # Slate-200 grid lines
    "text":     "#0f172a",   # Slate-900 primary text
    "text_dim": "#64748b",   # Slate-500 secondary text
    "accent":   "#0284c7",   # Sky-600 telemetry accent
    "accent_2": "#d97706",   # Amber-600 force accent
    "ok":       "#059669",   # Emerald-600 nominal/pass
    "info":     "#2563eb",   # Royal blue
    "warn":     "#d97706",   # Amber warning
    "crit":     "#dc2626",   # Crimson critical
    "violet":   "#7c3aed",   # Purple spectral/degradation
    "magenta":  "#c026d3",   # Sensor marker highlight
}

LEVEL_COLORS = [THEME["ok"], THEME["accent"], THEME["warn"], THEME["crit"]]

CMAP_NAMES = ["deflection", "stress", "diverging", "spectral"]

def get_cmap(name: str):
    if name == "deflection":
        return LinearSegmentedColormap.from_list(
            "custom_deflection", ["#2563eb", "#38bdf8", "#e2e8f0", "#f87171", "#dc2626"]
        )
    elif name == "stress":
        return LinearSegmentedColormap.from_list(
            "custom_stress", ["#0284c7", "#f8fafc", "#ea580c"]
        )
    elif name == "diverging":
        return plt.get_cmap("RdBu_r")
    return plt.get_cmap("coolwarm")

def app_qss() -> str:
    return f"""
        QMainWindow {{ background-color: {THEME['bg']}; }}
        QWidget {{ color: {THEME['text']}; font-family: 'Segoe UI', 'Inter', sans-serif; }}
        QLabel {{ color: {THEME['text']}; background: transparent; }}
        QGroupBox {{
            color: {THEME['accent']}; font-weight: bold; font-size: 10px;
            border: 1px solid {THEME['border']}; border-radius: 6px;
            margin-top: 10px; padding-top: 10px; background-color: {THEME['panel']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            padding: 0 6px; left: 8px; background-color: {THEME['panel']};
        }}
        QComboBox, QSpinBox, QLineEdit {{
            background-color: {THEME['panel']}; color: {THEME['text']};
            border: 1px solid {THEME['border']}; border-radius: 4px;
            padding: 4px 6px; font-size: 10px; font-weight: 500;
        }}
        QPushButton {{
            background-color: {THEME['panel']}; color: {THEME['text']};
            border-radius: 4px; padding: 6px 10px; font-size: 10px;
            font-weight: bold; border: 1px solid {THEME['border']};
        }}
        QPushButton:hover {{ background-color: {THEME['panel_2']}; border-color: {THEME['accent']}; }}
        QTableWidget {{
            background-color: {THEME['panel']}; border: 1px solid {THEME['border']};
            gridline-color: {THEME['grid']}; font-size: 9.5px;
        }}
        QHeaderView::section {{
            background-color: {THEME['panel_2']}; color: {THEME['text_dim']};
            font-weight: bold; font-size: 9px; border: 1px solid {THEME['border']};
            padding: 4px;
        }}
        QScrollBar:vertical {{ background: {THEME['bg']}; width: 8px; }}
        QScrollBar::handle:vertical {{ background: {THEME['border']}; border-radius: 4px; }}
    """

def btn_style(bg: str) -> str:
    return f"""
        QPushButton {{
            background-color: {bg}; color: white; font-weight: bold;
            border-radius: 4px; padding: 6px; font-size: 10px; border: none;
        }}
        QPushButton:hover {{ background-color: {THEME['text']}; color: white; }}
    """