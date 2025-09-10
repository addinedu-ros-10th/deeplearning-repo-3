#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assistive Navigation — PyQt6 GUI
--------------------------------
Layout and interactions matching the provided mockup.

Features (v1):
- Two video views (videoMain = D435 RGB or Cam0, videoRoad = Webcam Cam1 or Cam0)
- Alerts panel, Haptics panel (L/R meters + test), Audio panel (TTS queue + volume)
- Params panel (safe radius, class filter, mode), Connection panel, Log table
- Status bar with hints and FPS
- Optional RealSense D435 capture (aligned color + depth) if pyrealsense2 available
- Optional demo overlay (fake detections + distances) to validate UI without models

Next steps (plug-ins):
- Replace DemoDetector with YOLO-based detector (Ultralytics) and depth sampling
- Wire /haptics/* to real actuators over serial/GPIO
- Add TTS (ko-KR) via pyttsx3 or system TTS

Run:
  python assistive_nav_gui.py --demo  # default demo mode
  python assistive_nav_gui.py --realsense             # use D435 color as main
  python assistive_nav_gui.py --cam0 0 --cam1 1       # choose webcams
  python assistive_nav_gui.py --yolo /path/best.pt    # (stub) future use

Taemin’s notes:
- ObjectNames match our spec: videoMain, videoRoad, alertsPanel, hapticsPanel, audioPanel,
  paramsPanel, connectionPanel, logTable, statusBar.
- Minimal external deps: PyQt6, numpy, opencv-python. RealSense optional.
"""

from __future__ import annotations
import sys, time, argparse, threading, math, random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QSize
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QGridLayout, QVBoxLayout,
    QHBoxLayout, QGroupBox, QListWidget, QPushButton, QProgressBar,
    QSlider, QCheckBox, QRadioButton, QStatusBar, QFrame, QTableWidget,
    QTableWidgetItem, QSizePolicy
)

# ======================
# Helpers
# ======================

def np_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None:
        return QPixmap()
    if len(bgr.shape) == 2:
        h, w = bgr.shape
        bytes_per_line = w
        img = QImage(bgr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(img.copy())
    h, w, ch = bgr.shape
    rgb = bgr[..., ::-1].copy()
    bytes_per_line = ch * w
    img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())

class Lamp(QLabel):
    """Small circular status lamp."""
    def __init__(self, diameter=14, color=QColor(200,200,200), parent=None):
        super().__init__(parent)
        self._diameter = diameter
        self._color = color
        self.setFixedSize(QSize(diameter, diameter))
    def setColor(self, c: QColor):
        self._color = c
        self.update()
    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.PenStyle.NoPen)
        d = self._diameter
        p.drawEllipse(0,0,d,d)
        p.end()

# ======================
# Video Workers
# ======================

class CamWorker(QThread):
    frameReady = pyqtSignal(np.ndarray)
    def __init__(self, index: int = 0, width=640, height=480, fps=30, parent=None):
        super().__init__(parent)
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._running = True
        self._cap = None
    def stop(self):
        self._running = False
    def run(self):
        import cv2
        self._cap = cv2.VideoCapture(self.index)
        if self.width>0 and self.height>0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps>0:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self.frameReady.emit(frame)
        if self._cap:
            self._cap.release()

class RealSenseWorker(QThread):
    frameReady = pyqtSignal(np.ndarray, object, float)  # color, depth_frame, depth_scale
    def __init__(self, width=640, height=480, fps=30, parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.fps = fps
        self._running = True
        self.pipeline = None
        self.align = None
    def stop(self):
        self._running = False
    def run(self):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            print('[RealSenseWorker] pyrealsense2 import failed:', e)
            return
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipeline.start(config)
        self.pipeline = pipeline
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        print(f"[RealSense] started | depth_scale={depth_scale}")
        while self._running:
            frames = pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue
            color_np = np.asanyarray(color.get_data())
            self.frameReady.emit(color_np, depth, depth_scale)
        try:
            pipeline.stop()
        except Exception:
            pass

# ======================
# Demo Detector (fake boxes to drive UI)
# ======================

@dataclass
class Det:
    cls: str
    conf: float
    xyxy: Tuple[int,int,int,int]
    dist_m: Optional[float] = None

class DemoDetector:
    names = ['Human','Car']
    def __init__(self):
        self._t = 0.0
    def infer(self, frame: np.ndarray, depth_frame=None, depth_scale: float = 0.001) -> List[Det]:
        h, w = frame.shape[:2]
        t = time.time()
        s = (math.sin(t*0.8)+1)/2
        # two moving boxes
        x1 = int(w*0.18)
        y1 = int(h*0.12)
        x2 = int(w*0.34)
        y2 = int(h*0.62)
        car_w = int(w*0.24)
        car_h = int(h*0.34)
        cx = int(w*0.45 + (s-0.5)*w*0.15)
        cy = int(h*0.28)
        car = (cx, cy, cx+car_w, cy+car_h)
        human = (x1, y1, x2, y2)
        # fake distance oscillates
        d1 = 1.6 + 0.3*math.sin(t*0.9)
        d2 = 3.0 + 0.4*math.cos(t*0.7)
        return [
            Det('Human', 0.86, human, d1),
            Det('Car', 0.91, car, d2)
        ]

# ======================
# VideoView Widget
# ======================

class VideoView(QLabel):
    def __init__(self, object_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setMinimumSize(480, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel{background:#EEF1F7;border:1px solid #DDE0E6;border-radius:8px;}")
        self._last = None
    def setFrame(self, bgr: np.ndarray):
        self._last = bgr
        self.setPixmap(np_to_qpixmap(bgr).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def overlay(self, dets: List[Det], fps: float = 0.0, center_info: Optional[str] = None):
        if self._last is None:
            return
        import cv2
        img = self._last.copy()
        # draw boxes
        for d in dets:
            x1,y1,x2,y2 = d.xyxy
            color = (33,150,243) if d.cls=='Human' else (255,140,0)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f"{d.cls} {d.dist_m:.1f} m" if d.dist_m is not None else d.cls
            cv2.putText(img, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        # center cross
        h, w = img.shape[:2]
        cx, cy = w//2, h//2
        cv2.drawMarker(img, (cx,cy), (0,180,120), cv2.MARKER_CROSS, 20, 2)
        if center_info:
            cv2.putText(img, center_info, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,120,90), 2, cv2.LINE_AA)
        self.setFrame(img)

# ======================
# Main Window
# ======================

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Assistive Navigation — GUI")
        self.resize(1400, 860)
        self._t0 = time.time(); self._frames = 0; self._fps = 0.0

        root = QWidget(); self.setCentralWidget(root)
        grid = QGridLayout(root); grid.setContentsMargins(16,16,16,10); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(12)

        # --- Main area containers
        mainBox = QGroupBox(); mainBox.setTitle("")
        rightBox = QGroupBox(); rightBox.setTitle("")
        mainBox.setStyleSheet("QGroupBox{border:1px solid #E1E5EB;border-radius:12px;background:#FFFFFF;}")
        rightBox.setStyleSheet("QGroupBox{border:1px solid #E1E5EB;border-radius:12px;background:#FFFFFF;}")
        grid.addWidget(mainBox, 0, 0)
        grid.addWidget(rightBox, 0, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)

        # ===== Left: Main area layout
        L = QGridLayout(mainBox); L.setContentsMargins(12,12,12,12); L.setHorizontalSpacing(12); L.setVerticalSpacing(12)
        self.videoMain = VideoView("videoMain")
        self.videoRoad = VideoView("videoRoad")
        L.addWidget(self.videoMain, 0, 0)
        L.addWidget(self.videoRoad, 0, 1)
        L.setColumnStretch(0, 1); L.setColumnStretch(1, 1)

        # bottom 3 panels
        self.alertsPanel = self._build_alerts_panel()
        self.hapticsPanel = self._build_haptics_panel()
        self.audioPanel = self._build_audio_panel()
        L.addWidget(self.alertsPanel, 1, 0)
        bottomRight = QWidget(); brLay = QHBoxLayout(bottomRight); brLay.setContentsMargins(0,0,0,0); brLay.setSpacing(12)
        brLay.addWidget(self.hapticsPanel, 1)
        brLay.addWidget(self.audioPanel, 1)
        L.addWidget(bottomRight, 1, 1)
        L.setRowStretch(0, 3); L.setRowStretch(1, 2)

        # ===== Right: Controls / Status
        R = QVBoxLayout(rightBox); R.setContentsMargins(12,12,12,12); R.setSpacing(12)
        self.paramsPanel = self._build_params_panel()
        self.connectionPanel = self._build_connection_panel()
        self.logTable = self._build_log_table()
        R.addWidget(self.paramsPanel)
        R.addWidget(self.connectionPanel)
        R.addWidget(self.logTable, 1)

        # Status bar
        sb = QStatusBar(); self.setStatusBar(sb)
        self.statusBar().showMessage("Status: READY   |   ESC to stop  •  R to record  •  H to haptic test")

        # Capture workers
        self.main_color = None; self.main_depth = None; self.depth_scale = 0.001
        self.detector = DemoDetector() if args.demo or not args.yolo else DemoDetector()  # TODO: replace

        if args.realsense:
            try:
                self.rsWorker = RealSenseWorker()
                self.rsWorker.frameReady.connect(self.on_rs_frame)
                self.rsWorker.start()
            except Exception as e:
                print('[WARN] RealSense not started:', e)
                self._start_cam_workers()
        else:
            self._start_cam_workers()

        # Timer to refresh overlays / FPS
        self.uiTimer = QTimer(self); self.uiTimer.timeout.connect(self._on_tick); self.uiTimer.start(33)

    # --- builders
    def _card(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #E6E9EF;border-radius:10px;background:#F8F9FC;margin-top:6px;} QGroupBox::title{subcontrol-origin: margin; left:8px; top:-8px; background:transparent; padding:0 4px;}")
        return g

    def _build_alerts_panel(self) -> QWidget:
        g = self._card("alertsPanel")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(8)
        self.alertList = QListWidget(); self.alertList.setStyleSheet("QListWidget{background:#FFFFFF;border:1px solid #E6E9EF;border-radius:8px;}")
        v.addWidget(self.alertList, 1)
        # seed with examples
        for level, msg in [("CRITICAL","카메라 연결 끊김 (depth)"),("HIGH","왼쪽 사람 1.4 m 접근"),("MED","횡단보도 인식 (0.92)")]:
            self.alertList.addItem(f"{level}  |  {msg}")
        return g

    def _build_haptics_panel(self) -> QWidget:
        g = self._card("hapticsPanel")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(10)
        self.pbLeft = QProgressBar(); self.pbRight = QProgressBar()
        for pb in (self.pbLeft, self.pbRight):
            pb.setRange(0, 255); pb.setValue(0)
            pb.setTextVisible(False)
            pb.setFixedHeight(18)
            pb.setStyleSheet("QProgressBar{background:#EBEEF3;border:1px solid #E1E5EB;border-radius:9px;} QProgressBar::chunk{background:#2196F3;border-radius:7px;}")
        # labels
        v.addWidget(QLabel("Left motor"))
        v.addWidget(self.pbLeft)
        v.addWidget(QLabel("Right motor"))
        v.addWidget(self.pbRight)
        # buttons
        btns = QHBoxLayout();
        self.btnTestL = QPushButton("Test Left"); self.btnTestR = QPushButton("Test Right")
        self.btnTestL.setStyleSheet("QPushButton{background:#2196F3;color:white;border:none;border-radius:8px;padding:6px 10px;}")
        self.btnTestR.setStyleSheet("QPushButton{background:#FF8C00;color:white;border:none;border-radius:8px;padding:6px 10px;}")
        self.btnTestL.clicked.connect(lambda: self._haptic_test('L'))
        self.btnTestR.clicked.connect(lambda: self._haptic_test('R'))
        btns.addWidget(self.btnTestL); btns.addWidget(self.btnTestR)
        v.addLayout(btns)
        return g

    def _build_audio_panel(self) -> QWidget:
        g = self._card("audioPanel")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(8)
        self.ttsQueue = QListWidget(); self.ttsQueue.setStyleSheet("QListWidget{background:#FFFFFF;border:1px solid #E6E9EF;border-radius:8px;}")
        for s in ["왼쪽 사람 1.4미터","횡단보도 인식되었습니다","바닥 요철 주의"]:
            self.ttsQueue.addItem(s)
        v.addWidget(self.ttsQueue, 1)
        # volume
        row = QHBoxLayout(); row.addWidget(QLabel("Volume")); self.volSlider = QSlider(Qt.Orientation.Horizontal)
        self.volSlider.setRange(0,100); self.volSlider.setValue(80)
        row.addWidget(self.volSlider, 1)
        v.addLayout(row)
        return g

    def _build_params_panel(self) -> QWidget:
        g = self._card("paramsPanel")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(8)
        # safe radius
        row1 = QHBoxLayout(); row1.addWidget(QLabel("Safe radius")); self.safeSlider = QSlider(Qt.Orientation.Horizontal)
        self.safeSlider.setRange(50, 400); self.safeSlider.setValue(150)
        row1.addWidget(self.safeSlider, 1); v.addLayout(row1)
        # class filter
        v.addWidget(QLabel("Class filter"))
        cf = QHBoxLayout(); self.cbHuman = QCheckBox("Human"); self.cbCar = QCheckBox("Car"); self.cbBicycle = QCheckBox("Bicycle")
        self.cbHuman.setChecked(True); self.cbCar.setChecked(True)
        cf.addWidget(self.cbHuman); cf.addWidget(self.cbCar); cf.addWidget(self.cbBicycle); cf.addStretch(1)
        v.addLayout(cf)
        # mode
        v.addWidget(QLabel("Mode"))
        row2 = QHBoxLayout(); self.rbOutdoor = QRadioButton("outdoor"); self.rbIndoor = QRadioButton("indoor"); self.rbOutdoor.setChecked(True)
        row2.addWidget(self.rbOutdoor); row2.addWidget(self.rbIndoor); row2.addStretch(1)
        v.addLayout(row2)
        return g

    def _build_connection_panel(self) -> QWidget:
        g = self._card("connectionPanel")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(8)
        # three lamps
        row1 = QHBoxLayout(); row1.addWidget(QLabel("D435 depth")); self.lampD435 = Lamp(color=QColor(76,175,80))
        row1.addWidget(self.lampD435); row1.addStretch(1)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("Webcam")); self.lampWebcam = Lamp(color=QColor(255,193,7))
        row2.addWidget(self.lampWebcam); row2.addStretch(1)
        row3 = QHBoxLayout(); row3.addWidget(QLabel("Server")); self.lampServer = Lamp(color=QColor(220,53,69))
        row3.addWidget(self.lampServer); row3.addStretch(1)
        v.addLayout(row1); v.addLayout(row2); v.addLayout(row3)
        self.btnReconnect = QPushButton("Reconnect"); self.btnReconnect.setStyleSheet("QPushButton{background:#2196F3;color:white;border:none;border-radius:8px;padding:6px 10px;}")
        v.addWidget(self.btnReconnect, alignment=Qt.AlignmentFlag.AlignRight)
        return g

    def _build_log_table(self) -> QWidget:
        g = self._card("logTable")
        v = QVBoxLayout(g); v.setContentsMargins(10,8,10,10); v.setSpacing(6)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["time","event","action"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setStyleSheet("QTableWidget{background:#FFFFFF;border:1px solid #E6E9EF;border-radius:8px;}")
        self.tbl.setColumnWidth(0, 100); self.tbl.setColumnWidth(1, 180)
        v.addWidget(self.tbl)
        return g

    # --- capture wiring
    def _start_cam_workers(self):
        # main feed uses cam0; road feed uses cam1 (falls back to cam0)
        self.cam0 = CamWorker(self.args.cam0)
        self.cam0.frameReady.connect(self.on_cam0)
        self.cam0.start()
        if self.args.cam1 is None:
            self.cam1 = None
        else:
            self.cam1 = CamWorker(self.args.cam1)
            self.cam1.frameReady.connect(self.on_cam1)
            self.cam1.start()

    # --- slots
    def on_rs_frame(self, color_np, depth_frame, depth_scale: float):
        self.main_color = color_np
        self.main_depth = depth_frame
        self.depth_scale = depth_scale
        self.videoMain.setFrame(color_np)
        self._frames += 1
    def on_cam0(self, frame: np.ndarray):
        if self.args.realsense:
            # cam0 used as road if rs is main
            self.videoRoad.setFrame(frame)
        else:
            self.main_color = frame
            self.videoMain.setFrame(frame)
            self._frames += 1
    def on_cam1(self, frame: np.ndarray):
        self.videoRoad.setFrame(frame)

    def _haptic_test(self, side: str):
        # simple visual effect
        val = 220
        if side=='L':
            self.pbLeft.setValue(val)
        else:
            self.pbRight.setValue(val)
        # decay after 600ms
        QTimer.singleShot(600, lambda: (self.pbLeft.setValue(0) if side=='L' else self.pbRight.setValue(0)))
        self._append_log(f"Haptic {side}", "test")

    def _append_log(self, event: str, action: str):
        row = self.tbl.rowCount(); self.tbl.insertRow(row)
        now = time.strftime('%H:%M:%S')
        self.tbl.setItem(row, 0, QTableWidgetItem(now))
        self.tbl.setItem(row, 1, QTableWidgetItem(event))
        self.tbl.setItem(row, 2, QTableWidgetItem(action))
        self.tbl.scrollToBottom()

    def _on_tick(self):
        # FPS
        dt = time.time() - self._t0
        if dt >= 0.5:
            self._fps = self._frames/dt
            self._frames = 0; self._t0 = time.time()
        # Detection overlay (demo)
        if self.main_color is not None:
            dets = self.detector.infer(self.main_color, self.main_depth, self.depth_scale)
            safe_m = self.safeSlider.value()/100.0
            # haptic intensity demo: proportional inverse to distance
            for d in dets:
                if d.cls=='Human':
                    inten = int(max(0, min(255, 255*(safe_m/max(d.dist_m, 0.2)) )))
                    self.pbLeft.setValue(inten if ( (d.xyxy[0]+d.xyxy[2])//2 ) < (self.main_color.shape[1]//2) else 0)
                    self.pbRight.setValue(inten if ( (d.xyxy[0]+d.xyxy[2])//2 ) >= (self.main_color.shape[1]//2) else 0)
            center_info = f"FPS: {self._fps:.1f} | Safe {safe_m:.1f} m"
            self.videoMain.overlay(dets, fps=self._fps, center_info=center_info)
        # status bar refresh
        self.statusBar().showMessage(f"Status: READY   |   latency ~{int(1000/ max(self._fps,0.1))} ms   |   FPS {self._fps:.1f}   |   ESC to stop  •  R record  •  H haptic test")

    # --- key handling
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        elif e.key() == Qt.Key.Key_H:
            self._haptic_test('L')
            self._haptic_test('R')
        elif e.key() == Qt.Key.Key_R:
            self._append_log('Record', 'toggle')
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        try:
            if hasattr(self, 'cam0') and self.cam0: self.cam0.stop(); self.cam0.wait(500)
            if hasattr(self, 'cam1') and self.cam1: self.cam1.stop(); self.cam1.wait(500)
            if hasattr(self, 'rsWorker') and self.rsWorker: self.rsWorker.stop(); self.rsWorker.wait(500)
        finally:
            e.accept()

# ======================
# main
# ======================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true', help='use demo detector (default)')
    ap.add_argument('--realsense', action='store_true', help='use RealSense D435 as main feed')
    ap.add_argument('--cam0', type=int, default=0, help='index for main/road webcam #1')
    ap.add_argument('--cam1', type=int, default=None, help='index for road webcam #2 (optional)')
    ap.add_argument('--yolo', type=str, default=None, help='(future) path to YOLO .pt model')
    return ap.parse_args()

def main():
    args = parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
