#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, traceback, re
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QLineEdit, QSpinBox, QFileDialog, QInputDialog
)
from PyQt6.QtCore import Qt, QUrl, QObject, QEvent, QRect

# ---------- DB settings (env override supported) ----------
DB_HOST = os.getenv("BHC_DB_HOST", "database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com")
DB_PORT = int(os.getenv("BHC_DB_PORT", "3306"))
DB_USER = os.getenv("BHC_DB_USER", "robot")
DB_PASS = os.getenv("BHC_DB_PASS", "0310")  # consider using env var in production
DB_NAME = os.getenv("BHC_DB_NAME", "bhc_database")

UI_FILE = os.path.abspath("bha_gui.ui")

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

# ---------- Embedded video player (inside a container widget like videoRoad/videoMain) ----------
class _ResizeRelay(QObject):
    """Relays parent container resize events to keep video widget fitted."""
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._callback()
        return False

class EmbeddedVideoPlayer:
    def __init__(self, container_widget):
        """
        container_widget: e.g., self.videoRoad or self.videoMain (QLabel in .ui)
        """
        self.container = container_widget
        # Import multimedia lazily to give better error if missing
        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtMultimediaWidgets import QVideoWidget
        except Exception as e:
            raise RuntimeError(
                "PyQt6 멀티미디어 모듈이 없습니다. `pip install PyQt6 PyQt6-Qt6` 또는 배포 패키지에 포함해 주세요.\n"
                f"원인: {e}"
            )

        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtMultimediaWidgets import QVideoWidget

        # Video output widget that sits on top of the QLabel
        self.videoWidget = QVideoWidget(self.container)
        self.videoWidget.setObjectName("embeddedVideoWidget")
        self.videoWidget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videoWidget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.videoWidget.show()

        # Player and audio
        self.audio = QAudioOutput(self.container)
        self.player = QMediaPlayer(self.container)
        self.player.setVideoOutput(self.videoWidget)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)

        # Fit video widget to container and keep it fitted on resize
        self._relay = _ResizeRelay(self._fit_to_container)
        self.container.installEventFilter(self._relay)
        self._fit_to_container()

    def _fit_to_container(self):
        # Fill the container's rect
        r: QRect = self.container.rect()
        self.videoWidget.setGeometry(r)

    def play(self, source: QUrl):
        self.player.setSource(source)
        self.player.play()

    def stop(self):
        self.player.stop()

# ---------- Embedded camera (QCamera -> videoMain/road container) ----------
class EmbeddedCamera:
    """
    QCamera를 QLabel(컨테이너) 위에 올린 QVideoWidget으로 보여주는 래퍼.
    """
    def __init__(self, container_widget):
        self.container = container_widget
        try:
            from PyQt6.QtMultimedia import (
                QCamera, QMediaDevices, QMediaCaptureSession
            )
            from PyQt6.QtMultimediaWidgets import QVideoWidget
        except Exception as e:
            raise RuntimeError(
                "PyQt6 멀티미디어 모듈이 없습니다. `pip install PyQt6 PyQt6-Qt6` 설치가 필요합니다.\n"
                f"원인: {e}"
            )

        self.QCamera = QCamera
        self.QMediaDevices = QMediaDevices
        self.QMediaCaptureSession = QMediaCaptureSession
        self.QVideoWidget = QVideoWidget

        # 비디오 위젯을 컨테이너 전체로 깔기
        self.videoWidget = QVideoWidget(self.container)
        self.videoWidget.setObjectName("embeddedCameraWidget")
        self.videoWidget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videoWidget.show()

        # 크기 맞춤 리스너
        self._relay = _ResizeRelay(self._fit_to_container)
        self.container.installEventFilter(self._relay)
        self._fit_to_container()

        # 캡처 세션 및 카메라
        self.capture = self.QMediaCaptureSession(self.container)
        self.capture.setVideoOutput(self.videoWidget)

        self.camera = None

    def _fit_to_container(self):
        r: QRect = self.container.rect()
        self.videoWidget.setGeometry(r)

    def start(self, device_index: int = 0):
        """
        device_index: 0, 1, ... (기본 0번 카메라)
        """
        devices = self.QMediaDevices.videoInputs()
        if not devices:
            raise RuntimeError("사용 가능한 비디오 입력 장치가 없습니다. (웹캠 미검출)")

        if device_index < 0 or device_index >= len(devices):
            device_index = 0

        selected = devices[device_index]
        # 기존 카메라 정리
        if self.camera:
            try: self.camera.stop()
            except Exception: pass
            self.camera.deleteLater()
            self.camera = None

        self.camera = self.QCamera(selected)
        self.capture.setCamera(self.camera)
        self.camera.start()

    def stop(self):
        try:
            if self.camera:
                self.camera.stop()
        except Exception:
            pass

# ---------- Helpers ----------
def gdrive_to_direct(url_or_id: str) -> str:
    """
    Accepts a Google Drive share link or file id and returns a direct playable URL.
    Supported examples:
      - https://drive.google.com/file/d/<ID>/view?usp=sharing
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
      - <ID> (bare id)
    Returns: https://drive.google.com/uc?export=download&id=<ID>
    """
    s = url_or_id.strip()
    # Bare id?
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        file_id = s
    else:
        # Try to extract id from common patterns
        m = re.search(r"/file/d/([A-Za-z0-9_-]+)", s)
        if not m:
            m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
        if not m:
            m = re.search(r"[?&]export=download&id=([A-Za-z0-9_-]+)", s)
        if not m:
            # Not a drive url; return as-is (maybe it's some http(s) url to mp4)
            return s
        file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# ---------- Main Window loader ----------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)
        self.statusBar.showMessage("Status: READY   |   ESC to stop")

        # Demo list fill
        if hasattr(self, "alertList"):
            self.alertList.addItems([
                "CRITICAL  |  카메라 연결 끊김 (depth)",
                "HIGH      |  왼쪽 사람 1.4 m 접근",
                "MED       |  횡단보도 인식 (0.92)",
            ])
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)

        # VQA dialog
        if hasattr(self, "btnVqaLog"):
            self.btnVqaLog.clicked.connect(self.open_vqa_log)

        # --- Load video wiring (expects a QPushButton named btnLoad in the .ui) ---
        if hasattr(self, "btnLoad"):
            self.btnLoad.clicked.connect(self.on_btnLoad_clicked)

        # Prepare embedded players (camera to videoMain, file player to videoRoad)
        self._player_target = None
        self._cam = None
        try:
            # camera target: prefer videoMain, else fallback to videoRoad
            cam_target = getattr(self, "videoMain", None)
            if not isinstance(cam_target, QLabel):
                cam_target = getattr(self, "videoRoad", None)

            # file player target: prefer videoRoad, else fallback to videoMain
            player_target = getattr(self, "videoRoad", None)
            if not isinstance(player_target, QLabel):
                player_target = getattr(self, "videoMain", None)

            if isinstance(player_target, QLabel):
                self._player_target = EmbeddedVideoPlayer(player_target)
                player_target.setText("")

            if isinstance(cam_target, QLabel):
                self._cam = EmbeddedCamera(cam_target)
                cam_target.setText("")
                try:
                    # 기본 0번 카메라 시작
                    self._cam.start(device_index=1)
                    self.statusBar.showMessage("Camera: started on device 0")
                except Exception as ce:
                    QMessageBox.warning(self, "Camera", f"카메라 시작 실패: {ce}")
        except Exception as e:
            QMessageBox.warning(self, "Multimedia",
                                f"비디오/카메라 초기화 실패: {e}\n"
                                "영상은 외부 창에서 재생될 수 있습니다.")

        # (선택) 카메라 시작/정지 버튼 연결
        if hasattr(self, "btnCamStart") and self._cam:
            self.btnCamStart.clicked.connect(lambda: self._cam.start(0))
        if hasattr(self, "btnCamStop") and self._cam:
            self.btnCamStop.clicked.connect(self._cam.stop)

    def open_vqa_log(self):
        try:
            self.vqaDlg.close()
        except Exception:
            pass
        self.vqaDlg = VqaLogDialog(self)
        self.vqaDlg.setModal(False)
        self.vqaDlg.show()

    # --- Load video handler ---
    def on_btnLoad_clicked(self):
        """
        1) 사용자가 Google Drive 링크 또는 로컬 경로를 입력하면
        2) 가능한 경우 Drive 링크를 직접 재생 가능한 URL로 변환
        3) 지정된 비디오 영역에 임베드하여 재생
        """
        # 간단 입력창
        text, ok = QInputDialog.getText(
            self, "Load Video",
            "Google Drive 링크(또는 파일 ID)나 로컬 파일 경로를 입력하세요.\n"
            "비워두면 파일 선택 창이 열립니다."
        )
        if not ok:
            return

        if text.strip():
            source_str = text.strip()
            # 로컬 파일인지 URL/ID인지 판단
            if os.path.exists(source_str):
                url = QUrl.fromLocalFile(os.path.abspath(source_str))
            else:
                direct = gdrive_to_direct(source_str)
                url = QUrl(direct)
        else:
            # 파일 선택
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm);;All Files (*)"
            )
            if not path:
                return
            url = QUrl.fromLocalFile(os.path.abspath(path))

        # 재생
        try:
            if self._player_target is None:
                # Fallback: create a temporary window with a video widget
                self._player_target = self._create_floating_player()
            self._player_target.play(url)
            self.statusBar.showMessage(f"Playing: {url.toString()}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Play Error", f"영상을 재생하지 못했습니다:\n{e}")

    def _create_floating_player(self):
        """외부 창으로 재생하는 간단한 플레이어 (컨테이너 사용 불가 시)"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Video Player")
        dlg.resize(960, 540)
        lay = QVBoxLayout(dlg); lay.setContentsMargins(0,0,0,0)
        # 내부에 컨테이너 라벨을 하나 두고 임베드 플레이어를 장착
        container = QLabel(dlg); container.setText("")
        lay.addWidget(container)
        player = EmbeddedVideoPlayer(container)
        dlg.setModal(False); dlg.show()
        # 작은 헬퍼 객체를 돌려서 Main이 같은 인터페이스로 play/stop 호출하도록
        class _Shim:
            def __init__(self, p, d): self._p, self._d = p, d
            def play(self, url): self._p.play(url); self._d.show()
            def stop(self): self._p.stop()
        return _Shim(player, dlg)

    def keyPressEvent(self, e):
        if e.key()==Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(e)

if __name__=='__main__':
    # Ensure PyMySQL is available hint
    try:
        import pymysql  # noqa
    except Exception:
        print("TIP: install DB driver -> pip install pymysql")

    # Ensure PyQt multimedia is available hint
    try:
        from PyQt6.QtMultimedia import QMediaPlayer  # noqa
        from PyQt6.QtMultimediaWidgets import QVideoWidget  # noqa
        from PyQt6.QtMultimedia import QCamera  # noqa
    except Exception:
        print("TIP: install multimedia -> pip install PyQt6 PyQt6-Qt6")
        print("TIP: Ubuntu 계열은 gstreamer 플러그인과 v4l2 권한이 필요할 수 있습니다.")

    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
