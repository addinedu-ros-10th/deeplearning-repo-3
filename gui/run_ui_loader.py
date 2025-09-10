#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
import sys, time
from datetime import datetime

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('assistive_navigation_gui.ui', self)
        # seed sample data
        self.alertList.addItems([
            "CRITICAL  |  카메라 연결 끊김 (depth)",
            "HIGH      |  왼쪽 사람 1.4 m 접근",
            "MED       |  횡단보도 인식 (0.92)",
        ])
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["time","event","action"])
        self.tbl.verticalHeader().setVisible(False)
        self.volSlider.setValue(80)
        self.safeSlider.setValue(150)
        # status bar
        self.statusBar.showMessage("Status: READY   |   ESC to stop  •  H haptic test")
        # demo updates
        self.t = QTimer(self); self.t.timeout.connect(self.tick); self.t.start(500)
        self.pbLeft.setRange(0,255); self.pbRight.setRange(0,255)

    def tick(self):
        now = datetime.now().strftime("%H:%M:%S")
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, self._mkItem(now))
        self.tbl.setItem(r, 1, self._mkItem("demo heartbeat"))
        self.tbl.setItem(r, 2, self._mkItem("ok"))
        self.tbl.scrollToBottom()
        # haptic bars demo
        self.pbLeft.setValue((self.pbLeft.value()+40)%256)
        self.pbRight.setValue((self.pbRight.value()+140)%256)

    def _mkItem(self, text):
        from PyQt6.QtWidgets import QTableWidgetItem
        return QTableWidgetItem(text)

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

if __name__=='__main__':
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())
