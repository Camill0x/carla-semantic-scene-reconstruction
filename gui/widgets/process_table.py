from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class ProcessTable(QTableWidget):
    HEADERS = ["Process", "Status", "PID", "Last Exit"]

    def __init__(self) -> None:
        """Build the process table used by the inspector dialog."""
        super().__init__(0, len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

    def update_rows(self, rows: List[Dict[str, str]]) -> None:
        """Refresh the process table contents from the provided status rows."""
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [row.get("title", row["name"]), row["status"], row["pid"], row["last_exit"]]
            for col_index, value in enumerate(values):
                item = self.item(row_index, col_index)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(row_index, col_index, item)
                item.setText(value)
                if col_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row["name"])

            status_item = self.item(row_index, 1)
            if status_item is not None:
                if row["status"].startswith("Running"):
                    status_item.setForeground(QColor(Qt.GlobalColor.green))
                elif row["status"].startswith("Closed"):
                    status_item.setForeground(QColor("#ffa500"))
                else:
                    status_item.setForeground(QColor(Qt.GlobalColor.gray))
