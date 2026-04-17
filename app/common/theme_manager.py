from __future__ import annotations

from typing import Dict

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget
from qfluentwidgets import isDarkTheme, setTheme, setThemeColor

from app.common.config import cfg


def _to_hex(color: QColor | str, fallback: str) -> str:
    if isinstance(color, QColor):
        return color.name(QColor.HexRgb)

    if isinstance(color, str) and color.strip():
        q = QColor(color.strip())
        if q.isValid():
            return q.name(QColor.HexRgb)

    return fallback


def _theme_defaults(dark: bool) -> Dict[str, str]:
    if dark:
        return {
            "window_bg": "#1E1E1E",
            "panel_bg": "#252526",
            "card_bg": "#2D2D30",
            "border": "#3C3C3C",
            "text": "#D4D4D4",
            "muted_text": "#9DA1A6",
            "accent": "#007ACC",
            "accent_hover": "#1F8AD2",
        }

    return {
        "window_bg": "#F3F3F3",
        "panel_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "border": "#E1E1E1",
        "text": "#1F1F1F",
        "muted_text": "#616161",
        "accent": "#007ACC",
        "accent_hover": "#0062A3",
    }


def get_theme_palette() -> Dict[str, str]:
    dark = isDarkTheme()
    defaults = _theme_defaults(dark)

    accent_color = cfg.get(cfg.themeColor)
    accent = _to_hex(accent_color, defaults["accent"])

    return {
        "window_bg": _to_hex(cfg.ui_window_bg.value, defaults["window_bg"]),
        "panel_bg": _to_hex(cfg.ui_panel_bg.value, defaults["panel_bg"]),
        "card_bg": _to_hex(cfg.ui_card_bg.value, defaults["card_bg"]),
        "border": _to_hex(cfg.ui_border_color.value, defaults["border"]),
        "text": _to_hex(cfg.ui_text_color.value, defaults["text"]),
        "muted_text": defaults["muted_text"],
        "accent": accent,
        "accent_hover": defaults["accent_hover"],
        "is_dark": dark,
    }


def build_global_stylesheet() -> str:
    p = get_theme_palette()
    return f"""
QWidget {{
    color: {p['text']};
}}

QWidget#HomeInterface,
QWidget#AutoShortsInterface,
QWidget#TranscriptionInterface,
QWidget#TaskCreationInterface,
QWidget#SubtitleInterface,
QWidget#VideoSynthesisInterface,
QWidget#BatchProcessInterface {{
    background: {p['window_bg']};
}}

QWidget#settingInterface,
QWidget#scrollWidget {{
    background: transparent;
}}

QFrame,
CardWidget,
SettingCard,
QTableWidget,
QListWidget,
QTextEdit,
QPlainTextEdit {{
    background: {p['card_bg']};
    border: 1px solid {p['border']};
    border-radius: 8px;
}}

QHeaderView::section {{
    background: {p['panel_bg']};
    border: 1px solid {p['border']};
    padding: 4px;
}}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {{
    background: {p['panel_bg']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 4px;
}}

QPushButton {{
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 5px 10px;
}}

QPushButton:hover {{
    border-color: {p['accent']};
}}
"""


def _refresh_widget_tree(widget: QWidget):
    for method_name in ("refresh_theme", "on_theme_changed", "_apply_theme_style"):
        method = getattr(widget, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    for child in widget.findChildren(QWidget):
        for method_name in ("refresh_theme", "on_theme_changed"):
            method = getattr(child, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass


def apply_vscode_theme(refresh_widgets: bool = True):
    setTheme(cfg.get(cfg.themeMode))
    setThemeColor(cfg.get(cfg.themeColor))

    app = QApplication.instance()
    if app:
        app.setStyleSheet(build_global_stylesheet())
        if refresh_widgets:
            for widget in app.topLevelWidgets():
                _refresh_widget_tree(widget)
