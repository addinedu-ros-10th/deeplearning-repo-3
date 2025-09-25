#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, traceback, re, shutil, datetime, subprocess
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QLineEdit, QSpinBox, QFileDialog
)
from PyQt6.QtCore import Qt, QUrl, QObject, QEvent, QRect

# ---------- DB settings ----------
DB_HOST = os.getenv("BHC_DB_HOST", "database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com")
DB_PORT = int(os.getenv("BHC_DB_PORT", "3306"))
DB_USER = os.getenv("BHC_DB_USER", "robot")
DB_PASS = os.getenv("BHC_DB_PASS", "0310")
DB_NAME = os.getenv("BHC_DB_NAME", "bhc_database")

UI_FILE = os.path.abspath("bha_gui.ui")

# ---------- Drive/rclone settings (필요 시 수정) ----------
RCLONE_REMOTE   = os.getenv("RCLONE_REMOTE", "dl_project")                  # rclone 원격 이름
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1k2mk0qF2i6Zw1fUg0YBNgBMY9OL5EkBI")  # 대상 폴더 ID

# ---------- VQA Log dialog ----------
class VqaLogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VQA Log Viewer")
        self.resize(720, 520)
        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(8)

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

        self.tbl = QTableWidget(0, 3, self)
        self.tbl.setHorizontalHeaderLabels(["question", "answer", "created_at"])
        self.tbl.verticalHeader().setVisible(False)
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.tbl, 1)

        self.status = QLabel("Ready")
        root.addWidget(self.status)

        self.btnRefresh.clicked.connect(self.load_data)
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

# ---------- Embedded video player ----------
class _ResizeRelay(QObject):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._callback()
        return False

class EmbeddedVideoPlayer:
    def __init__(self, container_widget):
        self.container = container_widget
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtMultimediaWidgets import QVideoWidget

        self.videoWidget = QVideoWidget(self.container)
        self.videoWidget.setObjectName("embeddedVideoWidget")
        self.videoWidget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videoWidget.show()

        self.audio = QAudioOutput(self.container)
        self.player = QMediaPlayer(self.container)
        self.player.setVideoOutput(self.videoWidget)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.8)

        self._relay = _ResizeRelay(self._fit_to_container)
        self.container.installEventFilter(self._relay)
        self._fit_to_container()

    def _fit_to_container(self):
        self.videoWidget.setGeometry(self.container.rect())

    def play(self, source: QUrl):
        self.player.setSource(source)
        self.player.play()

    def stop(self):
        self.player.stop()

# ---------- Embedded camera (with recorder) ----------
class EmbeddedCamera:
    """QLabel 컨테이너에 QVideoWidget 프리뷰 + QMediaRecorder 녹화"""
    def __init__(self, container_widget):
        self.container = container_widget
        from PyQt6.QtMultimedia import QCamera, QMediaDevices, QMediaCaptureSession, QMediaRecorder, QMediaFormat
        from PyQt6.QtMultimediaWidgets import QVideoWidget

        self.QCamera = QCamera
        self.QMediaDevices = QMediaDevices
        self.QMediaCaptureSession = QMediaCaptureSession
        self.QMediaRecorder = QMediaRecorder
        self.QMediaFormat = QMediaFormat

        self.videoWidget = QVideoWidget(self.container)
        self.videoWidget.setObjectName("embeddedCameraWidget")
        self.videoWidget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.videoWidget.show()

        self._relay = _ResizeRelay(self._fit_to_container)
        self.container.installEventFilter(self._relay)
        self._fit_to_container()

        self.capture = self.QMediaCaptureSession(self.container)
        self.capture.setVideoOutput(self.videoWidget)

        self.camera = None
        self.recorder = self.QMediaRecorder(self.container)
        self.capture.setRecorder(self.recorder)

    def _fit_to_container(self):
        self.videoWidget.setGeometry(self.container.rect())

    def start(self, device_index: int = 1):
        devices = self.QMediaDevices.videoInputs()
        if not devices:
            raise RuntimeError("사용 가능한 비디오 입력 장치가 없습니다.")
        if device_index < 0 or device_index >= len(devices):
            device_index = 0
        selected = devices[device_index]

        if self.camera:
            try: self.camera.stop()
            except Exception: pass
            self.camera.deleteLater()
            self.camera = None

        self.camera = self.QCamera(selected)
        self.capture.setCamera(self.camera)
        self.camera.start()

    # ---- 녹화: AVI + MotionJPEG (호환성 우선) ----
    def start_recording(self, filepath: str):
        from PyQt6.QtMultimedia import QMediaFormat
        base, _ = os.path.splitext(filepath)
        filepath_avi = base + ".avi"   # 확장자 통일

        fmt = self.QMediaFormat()
        fmt.setFileFormat(QMediaFormat.FileFormat.AVI)
        fmt.setVideoCodec(QMediaFormat.VideoCodec.MotionJPEG)

        self.recorder.setMediaFormat(fmt)
        self.recorder.setQuality(self.QMediaRecorder.Quality.NormalQuality)
        self.recorder.setOutputLocation(QUrl.fromLocalFile(os.path.abspath(filepath_avi)))
        try:
            self.recorder.stop()
        except Exception:
            pass
        self.recorder.record()

    def stop_recording(self) -> str:
        self.recorder.stop()
        loc = self.recorder.actualLocation()
        return loc.toLocalFile() if loc.isLocalFile() else ""

# ---------- Helpers ----------
def gdrive_to_direct(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        file_id = s
    else:
        m = re.search(r"/file/d/([A-Za-z0-9_-]+)", s)
        if not m: m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
        if not m: return s
        file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def upload_via_rclone(local_path: str) -> str:
    """rclone로 팀드라이브(폴더 ID)로 업로드"""
    rclone = shutil.which("rclone")
    if not rclone:
        return "rclone 미설치: sudo apt-get install rclone 후 재시도하세요."

    remote = RCLONE_REMOTE
    folder_id = DRIVE_FOLDER_ID
    if not remote or not folder_id:
        return "rclone 원격 또는 폴더 ID 미설정(RCLONE_REMOTE/DRIVE_FOLDER_ID)."

    dest = f"{remote}:{os.path.basename(local_path)}"
    cmd = [rclone, "copyto", local_path, dest, f"--drive-root-folder-id={folder_id}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return f"rclone 업로드 성공 → {remote}:{os.path.basename(local_path)}"
        err = (res.stderr or res.stdout).strip()
        return f"rclone 업로드 실패: {err}"
    except Exception as e:
        return f"rclone 실행 오류: {e}"

def try_save_to_drive(local_path: str) -> str:
    """1) rclone 업로드 시도 → 2) 동기화 폴더 복사(가능 시)"""
    if not local_path or not os.path.exists(local_path):
        return "녹화 파일이 존재하지 않습니다."

    # 1) rclone 우선
    msg_rc = upload_via_rclone(local_path)
    if msg_rc.startswith("rclone 업로드 성공"):
        return msg_rc

    # 2) 로컬 동기화 폴더(있을 때)
    candidates = [
        "/home/addinedu/GoogleDrive/프로젝트녹화",
        os.getenv("GOOGLE_DRIVE_SYNC_DIR"),
        os.path.expanduser("~/Google Drive"),
        os.path.expanduser("~/GoogleDrive"),
        os.path.expanduser("~/내 드라이브"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            try:
                dst = os.path.join(c, os.path.basename(local_path))
                shutil.copy2(local_path, dst)
                return f"{msg_rc} | 동기화 폴더 복사: {dst}"
            except Exception as e:
                return f"{msg_rc} | 동기화 폴더 복사 실패: {e}"

    return f"{msg_rc} | Drive 동기화 폴더가 없어 로컬에만 저장되었습니다."

# ---------- Custom LoadVideoDialog ----------
class LoadVideoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Video")
        self.setModal(True)
        lay = QVBoxLayout(self)
        label = QLabel(
            "Google Drive 링크(또는 파일 ID)나 로컬 파일 경로를 입력하세요.\n"
            "비워두면 파일 선택 창이 열립니다."
        )
        lay.addWidget(label)
        self.line = QLineEdit(self); lay.addWidget(self.line)
        row = QHBoxLayout()
        btnCancel = QPushButton("Cancel", self)
        btnOk = QPushButton("OK", self)
        row.addStretch(1); row.addWidget(btnCancel); row.addWidget(btnOk)
        lay.addLayout(row)
        btnOk.clicked.connect(self.accept)
        btnCancel.clicked.connect(self.reject)

    def getText(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.line.text().strip(), True
        return "", False

# ---------- Main Window ----------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)
        self.statusBar.showMessage("Status: READY   |   ESC to stop")

        self._loadDialogOpen = False
        self._floating_player = None
        self._recording = False
        self._record_dir = os.path.expanduser("~/Videos/bha_captures")
        os.makedirs(self._record_dir, exist_ok=True)

        if hasattr(self, "alertList"):
            self.alertList.addItems([
                "CRITICAL  |  카메라 연결 끊김 (depth)",
                "HIGH      |  왼쪽 사람 1.4 m 접근",
                "MED       |  횡단보도 인식 (0.92)",
            ])
        if hasattr(self, "tbl"):
            self.tbl.setRowCount(0)

        if hasattr(self, "btnVqaLog"):
            self.btnVqaLog.clicked.connect(self.open_vqa_log)

        if hasattr(self, "btnLoad"):
            try: self.btnLoad.clicked.disconnect()
            except Exception: pass
            self.btnLoad.clicked.connect(self.handle_btnLoad_clicked)

        self._player_target = None
        self._cam = None
        try:
            cam_target = getattr(self, "videoMain", None) or getattr(self, "videoRoad", None)
            player_target = getattr(self, "videoRoad", None) or getattr(self, "videoMain", None)

            if isinstance(player_target, QLabel):
                self._player_target = EmbeddedVideoPlayer(player_target)
                player_target.setText("")
            if isinstance(cam_target, QLabel):
                self._cam = EmbeddedCamera(cam_target)
                cam_target.setText("")
                try:
                    self._cam.start(device_index=1)
                    self.statusBar.showMessage("Camera: started on device 1 (USB)")
                except Exception as ce:
                    QMessageBox.warning(self, "Camera", f"카메라 시작 실패: {ce}")
        except Exception as e:
            QMessageBox.warning(self, "Multimedia", f"비디오/카메라 초기화 실패: {e}")

        if hasattr(self, "btnCamStart") and self._cam:
            self.btnCamStart.clicked.connect(lambda: self._cam.start(1))
        if hasattr(self, "btnCamStop") and self._cam:
            self.btnCamStop.clicked.connect(self._cam.stop)

        # ---- 녹화/정지 버튼: videoMain 아래 중앙 정렬 ----
        self._build_record_buttons_below_videoMain()
        self.installEventFilter(_ResizeRelay(self._position_record_buttons))

    def _build_record_buttons_below_videoMain(self):
        self.btnRecord = QPushButton("● Record", self); self.btnRecord.setStyleSheet("font-weight:600;")
        self.btnStopRec = QPushButton("■ Stop", self); self.btnStopRec.setEnabled(False)
        self.btnRecord.clicked.connect(self._on_record_clicked)
        self.btnStopRec.clicked.connect(self._on_stop_record_clicked)
        self.btnRecord.resize(self.btnRecord.sizeHint()); self.btnStopRec.resize(self.btnStopRec.sizeHint())
        self._position_record_buttons()

    def _position_record_buttons(self):
        vm = getattr(self, "videoMain", None)
        if not isinstance(vm, QLabel):
            vm = getattr(self, "videoRoad", None)
            if not isinstance(vm, QLabel):
                return
        g = vm.geometry(); margin = 6; spacing = 8
        if self.btnRecord.width() == 0: self.btnRecord.resize(self.btnRecord.sizeHint())
        if self.btnStopRec.width() == 0: self.btnStopRec.resize(self.btnStopRec.sizeHint())
        total_w = self.btnRecord.width() + spacing + self.btnStopRec.width()
        start_x = g.center().x() - total_w // 2
        y = g.bottom() + margin
        self.btnRecord.move(start_x, y)
        self.btnStopRec.move(start_x + self.btnRecord.width() + spacing, y)
        self.btnRecord.raise_(); self.btnStopRec.raise_()
        self.btnRecord.show(); self.btnStopRec.show()

    def open_vqa_log(self):
        try: self.vqaDlg.close()
        except Exception: pass
        self.vqaDlg = VqaLogDialog(self); self.vqaDlg.setModal(False); self.vqaDlg.show()

    def handle_btnLoad_clicked(self):
        if self._loadDialogOpen: return
        self._loadDialogOpen = True
        try:
            dlg = LoadVideoDialog(self)
            text, ok = dlg.getText()
            if not ok: return
            if text:
                url = QUrl.fromLocalFile(os.path.abspath(text)) if os.path.exists(text) else QUrl(gdrive_to_direct(text))
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm);;All Files (*)")
                if not path: return
                url = QUrl.fromLocalFile(os.path.abspath(path))
            if self._player_target is None:
                self._player_target = self._create_floating_player()
            self._player_target.play(url)
            self.statusBar.showMessage(f"Playing: {url.toString()}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Play Error", f"영상을 재생하지 못했습니다:\n{e}")
        finally:
            self._loadDialogOpen = False

    # ---------- Recording ----------
    def _on_record_clicked(self):
        if not self._cam or not self._cam.camera:
            QMessageBox.warning(self, "Record", "카메라가 시작되지 않았습니다."); return
        if self._recording: return
        filename = f"record_{now_ts()}.mp4"
        path = os.path.join(self._record_dir, filename)
        try:
            self._cam.start_recording(path)
            self._recording = True
            self.btnRecord.setEnabled(False); self.btnStopRec.setEnabled(True)
            self.statusBar.showMessage(f"Recording... -> {path}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Record", f"녹화 시작 실패: {e}")

    def _on_stop_record_clicked(self):
        if not self._recording: return
        try:
            saved = self._cam.stop_recording()
            self._recording = False
            self.btnRecord.setEnabled(True); self.btnStopRec.setEnabled(False)
            msg = f"Saved: {saved}" if saved else "Saved (path unknown)"
            drive_msg = try_save_to_drive(saved)
            self.statusBar.showMessage(f"{msg} | {drive_msg}")
            QMessageBox.information(self, "Record", f"{msg}\n{drive_msg}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Record", f"녹화 정지 실패: {e}")

    # ---------- Floating player ----------
    def _create_floating_player(self):
        if self._floating_player: return self._floating_player
        dlg = QDialog(self); dlg.setWindowTitle("Video Player"); dlg.resize(960, 540)
        lay = QVBoxLayout(dlg); lay.setContentsMargins(0,0,0,0)
        container = QLabel(dlg); container.setText(""); lay.addWidget(container)
        player = EmbeddedVideoPlayer(container); dlg.setModal(False)
        class _Shim:
            def __init__(self, p, d): self._p, self._d = p, d
            def play(self, url): self._p.play(url); self._d.show(); self._d.raise_(); self._d.activateWindow()
            def stop(self): self._p.stop()
        self._floating_player = _Shim(player, dlg); return self._floating_player

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(e)

# ---------- main ----------
if __name__ == '__main__':
    try: import pymysql
    except Exception: print("TIP: pip install pymysql")
    try:
        from PyQt6.QtMultimedia import QMediaPlayer, QCamera
        from PyQt6.QtMultimediaWidgets import QVideoWidget
    except Exception:
        print("TIP: pip install PyQt6 PyQt6-Qt6 (and gstreamer plugins)")

    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
