from typing import Callable

from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout


class NavigationCard(QFrame):
    def __init__(self, *, title: str, description: str, button_text: str, on_click: Callable[[], None]) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #9db0cc;")
        button = QPushButton(button_text)
        button.clicked.connect(on_click)
        layout.addWidget(title_label)
        layout.addWidget(desc_label, 1)
        layout.addWidget(button)
