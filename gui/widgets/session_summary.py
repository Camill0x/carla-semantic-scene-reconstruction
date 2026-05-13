from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class SessionSummary(QWidget):
    def __init__(self, specs: Optional[List[Tuple[str, str]]] = None) -> None:
        super().__init__()
        self.cards: Dict[str, QLabel] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        if specs is None:
            specs = [
                ("running", "Active Processes"),
                ("core", "Shared Memory"),
                ("detectors", "Active Detectors"),
                ("manual", "Manual Driving"),
            ]
        for index, (key, title) in enumerate(specs):
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            card.setProperty("summaryCard", True)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 14, 14, 14)
            title_label = QLabel(title)
            title_label.setProperty("summaryTitle", True)
            value_label = QLabel("-")
            value_label.setProperty("summaryValue", True)
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            layout.addWidget(card, 0, index)
            self.cards[key] = value_label

    def update_values(self, values: Dict[str, str]) -> None:
        for key, label in self.cards.items():
            label.setText(str(values.get(key, "-")))
