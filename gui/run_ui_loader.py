#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, traceback
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QLineEdit, QSpinBox
)
from PyQt6.QtCore import Qt

# ---------- DB settings (env override supported) ----------
DB_HOST = os.getenv("BHC_DB_HOST", "database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com")
DB_PORT = int(os.getenv("BHC_DB_PORT", "3306"))
DB_USER = os.getenv("BHC_DB_USER", "robot")
DB_PASS = os.getenv("BHC_DB_PASS", "0310")  # consider using env var in production
DB_NAME = os.getenv("BHC_DB_NAME", "bhc_database")

UI_FILE = os.path.abspath("assistive_navigation_gui.ui")

# ---------- VQA Log dialog ----------
class VqaLogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VQA Log Viewer")
        self.resize(720, 520)
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)

        # Header row: title + controls
        title_row = QHBoxLayout()
        title = QLabel(f"{DB_NAME}.vqa_log  (host: {DB_HOST}, user: {DB_USER})")
        title.setStyleSheet("font-weight:600;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("Limit"))
        self.limitSpin = QSpinBox(); self.limitSpin.setRange(1, 5000); self.limitSpin.setValue(500)
        title_row.addWidget(self.limitSpin)
        self.btnRefresh = QPushButton("Refresh"); title_row.addWidget(self.btnRefresh)
        root.addLayout(title_row)

        # Table
        self.tbl = QTableWidget(0, 3, self)
        self.tbl.setHorizontalHeaderLabels(["question", "answer", "created_at"])
        self.tbl.verticalHeader().setVisible(False)
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.tbl, 1)

        # Footer
        self.status = QLabel("Ready")
        root.addWidget(self.status)

        # Events
        self.btnRefresh.clicked.connect(self.load_data)

        # First load
        self.load_data()

    def load_data(self):
        limit = self.limitSpin.value()
        try:
            import pymysql
            conn = pymysql.connect(
                host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
                database=DB_NAME, charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
            )
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT question, answer, created_at FROM vqa_log ORDER BY id DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "DB Error", f"Failed to load data:\n{e}")
            self.status.setText("DB error")
            return

        self.tbl.setRowCount(0)
        for r in rows:
            row = self.tbl.rowCount(); self.tbl.insertRow(row)
            self.tbl.setItem(row, 0, QTableWidgetItem(str(r.get("question",""))))
            self.tbl.setItem(row, 1, QTableWidgetItem(str(r.get("answer",""))))
            self.tbl.setItem(row, 2, QTableWidgetItem(str(r.get("created_at",""))))
        self.status.setText(f"Loaded {len(rows)} row(s).")

# ---------- Main Window loader ----------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)
        self.statusBar.showMessage("Status: READY   |   ESC to stop")
        if hasattr(self, "alertList"):
            self.alertList.addItems([
                "CRITICAL  |  카메라 연결 끊김 (depth)",
                "HIGH      |  왼쪽 사람 1.4 m 접근",
                "MED       |  횡단보도 인식 (0.92)",
            ])
        if hasattr(self, "tbl"):  # log table heartbeat demo
            self.tbl.setRowCount(0)
        # bind new button
        if hasattr(self, "btnVqaLog"):
            self.btnVqaLog.clicked.connect(self.open_vqa_log)

    def open_vqa_log(self):
        try:
            self.vqaDlg.close()
        except Exception:
            pass
        self.vqaDlg = VqaLogDialog(self)
        self.vqaDlg.setModal(False)
        self.vqaDlg.show()

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(e)

if __name__=='__main__':
    # Ensure PyMySQL is available hint
    try:
        import pymysql  # noqa
    except Exception:
        print("TIP: install DB driver -> pip install pymysql")
    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
