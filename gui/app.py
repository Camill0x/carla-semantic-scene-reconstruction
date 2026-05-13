from PySide6.QtWidgets import QApplication

from gui.config import APP_NAME
from gui.widgets.main_window import MainWindow


def create_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            font-size: 13px;
            color: #e6edf3;
        }
        QMainWindow {
            background: #0d1117;
        }
        QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {
            background: transparent;
            border: none;
        }
        QSplitter {
            background: transparent;
        }
        QSplitter::handle {
            background: #111827;
        }
        QLabel {
            background: transparent;
        }
        QFrame[frameShape="6"], QGroupBox {
            border: 1px solid #1f2937;
            border-radius: 14px;
            background: #111827;
        }
        QPushButton {
            background: #2563eb;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #1d4ed8;
        }
        QPushButton[buttonRole="secondary"] {
            background: #1f2937;
            color: #e6edf3;
        }
        QPushButton[buttonRole="secondary"]:hover {
            background: #334155;
        }
        QPushButton[buttonRole="success"] {
            background: #16a34a;
            color: #ffffff;
        }
        QPushButton[buttonRole="success"]:hover {
            background: #15803d;
        }
        QPushButton[buttonRole="warning"] {
            background: #d97706;
            color: #ffffff;
        }
        QPushButton[buttonRole="warning"]:hover {
            background: #b45309;
        }
        QPushButton[buttonRole="danger"] {
            background: #dc2626;
        }
        QPushButton[buttonRole="danger"]:hover {
            background: #b91c1c;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QTableWidget, QListWidget, QTabWidget::pane {
            background: #0f172a;
            border: 1px solid #243041;
            border-radius: 8px;
        }
        QHeaderView::section {
            background: #172033;
            color: #d7e3f4;
            border: none;
            padding: 6px;
        }
        QListWidget::item {
            padding: 10px 8px;
            border-radius: 8px;
            margin: 2px;
        }
        QListWidget::item:selected {
            background: #1d4ed8;
            color: #ffffff;
        }
        QTabBar::tab {
            background: #172033;
            color: #d7e3f4;
            padding: 8px 14px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background: #2563eb;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: 700;
            margin-top: 10px;
            padding-top: 12px;
            background: #111827;
        }
        QGroupBox::title {
            left: 12px;
            padding: 0 4px;
        }
        QLabel[summaryTitle="true"] {
            color: #8ea0bd;
            font-size: 12px;
            font-weight: 600;
        }
        QLabel[summaryValue="true"] {
            font-size: 22px;
            font-weight: 700;
        }
        """)
    return app


def create_main_window() -> MainWindow:
    return MainWindow()
