#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, traceback, re, shutil, datetime, subprocess, socket, threading
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QLineEdit, QSpinBox, QFileDialog, QWidget
)
from PyQt6.QtCore import Qt, QUrl, QObject, QEvent, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

import numpy as np
import cv2

# =========================
# 서버(네트워크) 스트림 URL
# =========================
STREAM_URL = os.getenv("BHC_STREAM_URL", "http://192.168.0.155:8000/stream")

# ---------- DB settings ----------
DB_HOST = os.getenv("BHC_DB_HOST", "database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com")
DB_PORT = int(os.getenv("BHC_DB_PORT", "3306"))
DB_USER = os.getenv("BHC_DB_USER", "robot")
DB_PASS = os.getenv("BHC_DB_PASS", "0310")
DB_NAME = os.getenv("BHC_DB_NAME", "bhc_database")

UI_FILE = os.path.abspath("bha_gui.ui")

# ---------- Heartbeat dashboard thresholds ----------
ONLINE_SEC = 10
WARN_SEC   = 30

# ---------- Drive/rclone settings ----------
RCLONE_REMOTE   = os.getenv("RCLONE_REMOTE", "dl_project")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1XZv-AhjJysnndQMxY9QWJfC1vlub8zrU")

# ---------- Assets ----------
BHC_IMG_PATH = "/home/addinedu/dev_ws/dl_project/gui_asset/bhc.png"

# ---------- Utils ----------
class _ResizeRelay(QObject):
    def __init__(self, callback): super().__init__(); self._callback = callback
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize: self._callback()
        return False

def gdrive_to_direct(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        file_id = s
    else:
        m = re.search(r"/file/d/([A-Za-z0-9_-]+)", s) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
        if not m: return s
        file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def upload_via_rclone(local_path: str) -> str:
    rclone = shutil.which("rclone")
    if not rclone:
        return "rclone 미설치: sudo apt-get install rclone 후 재시도하세요."
    remote = RCLONE_REMOTE; folder_id = DRIVE_FOLDER_ID
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
    if not local_path or not os.path.exists(local_path):
        return "녹화 파일이 존재하지 않습니다."
    msg_rc = upload_via_rclone(local_path)
    if msg_rc.startswith("rclone 업로드 성공"):
        return msg_rc
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

# ---------- Heartbeat dashboard helpers ----------
def _led_widget(color_css: str, text: str = "") -> QWidget:
    w = QWidget(); dot = QLabel(); dot.setFixedSize(14, 14)
    dot.setStyleSheet(f"background:{color_css}; border-radius:7px;")
    txt = QLabel(text)
    lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
    lay.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(txt, 0, Qt.AlignmentFlag.AlignVCenter)
    return w

def _state_color(sec_since_seen: int, status: str) -> tuple[str, str]:
    level = "green" if sec_since_seen <= ONLINE_SEC else ("orange" if sec_since_seen <= WARN_SEC else "red")
    if status != "OK": level = "orange" if level == "green" else "red"
    label = {"green":"Online", "orange":"Warning", "red":"Offline"}[level]
    color = {"green":"#22c55e", "orange":"#f59e0b", "red":"#ef4444"}[level]
    return color, label

def _ensure_conn_table_on(panel: QWidget) -> QTableWidget:
    table = panel.findChild(QTableWidget, "connTable")
    if table: return table
    if panel.layout() is None:
        lay = QVBoxLayout(panel); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
    else:
        lay = panel.layout()
    table = QTableWidget(panel); table.setObjectName("connTable"); lay.addWidget(table)
    return table

def _init_conn_table(table: QTableWidget):
    table.setColumnCount(4)
    table.setHorizontalHeaderLabels(["Component", "IP", "Status", "Last Seen"])
    hh = table.horizontalHeader(); hh.setStretchLastSection(True)
    for i in range(4):
        hh.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
    table.setShowGrid(False); table.setAlternatingRowColors(True)
    table.setEditTriggers(table.EditTrigger.NoEditTriggers)
    table.setSelectionMode(table.SelectionMode.NoSelection)
    table.verticalHeader().setVisible(False)

def _fetch_heartbeat_rows():
    import pymysql
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", autocommit=True
    )
    with conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT component, ip, status, last_seen,
                   TIMESTAMPDIFF(SECOND, last_seen, NOW()) AS sec_since_seen
            FROM system_heartbeat
            ORDER BY component, ip
        """)
        rows = cur.fetchall()
    return rows

def _refresh_dashboard(table: QTableWidget):
    try:
        rows = _fetch_heartbeat_rows()
    except Exception as e:
        traceback.print_exc()
        table.setRowCount(1)
        table.setItem(0, 0, QTableWidgetItem("DB ERROR"))
        table.setItem(0, 1, QTableWidgetItem(str(e)))
        return
    table.setRowCount(len(rows))
    for r, (component, ip, status, last_seen, sec_since) in enumerate(rows):
        table.setItem(r, 0, QTableWidgetItem(str(component)))
        table.setItem(r, 1, QTableWidgetItem(str(ip)))
        color, label = _state_color(int(sec_since), str(status))
        table.setCellWidget(r, 2, _led_widget(color, label))
        table.setItem(r, 3, QTableWidgetItem(f"{last_seen}  (+{sec_since}s)"))
        table.setRowHeight(r, 26)

# ---------- Network Stream Worker (URL 수신 + 녹화) ----------
class NetworkStreamWorker(QThread):
    frameReady = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url
        self._running = False
        self._rec_lock = threading.Lock()
        self._recording = False
        self._writer = None
        self._record_path = None
        self._want_mp4 = True
        self._fps = 15.0
        self._size = None  # (w, h)

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # self.error.emit(f"스트림 연결 실패: {self.url}")
            return
        try:
            while self._running:
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if self._size is None:
                    h, w = frame_bgr.shape[:2]
                    self._size = (w, h)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 1:
                        self._fps = float(fps)
                with self._rec_lock:
                    if self._recording and self._writer is None and self._size is not None:
                        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if self._want_mp4 else "MJPG"))
                        path = self._record_path
                        if self._want_mp4 and not path.lower().endswith(".mp4"):
                            base, _ = os.path.splitext(path); path = base + ".mp4"; self._record_path = path
                        if (not self._want_mp4) and not path.lower().endswith(".avi"):
                            base, _ = os.path.splitext(path); path = base + ".avi"; self._record_path = path
                        writer = cv2.VideoWriter(path, fourcc, self._fps, self._size)
                        if not writer.isOpened():
                            self.error.emit("녹화기 초기화 실패(코덱/권한 확인).")
                            self._recording = False
                        else:
                            self._writer = writer
                    if self._recording and self._writer is not None:
                        try:
                            self._writer.write(frame_bgr)
                        except Exception as we:
                            self.error.emit(f"녹화 중 오류: {we}")
                            try: self._writer.release()
                            except Exception: pass
                            self._writer = None
                            self._recording = False
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frameReady.emit(qimg.copy())
        except Exception as e:
            self.error.emit(f"스트림 루프 오류: {e}")
        finally:
            try:
                if self._writer is not None: self._writer.release()
            except Exception:
                pass
            cap.release()

    def start_recording(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        self._want_mp4 = (ext == ".mp4")
        with self._rec_lock:
            self._record_path = filepath
            self._recording = True
        return True

    def stop_recording(self) -> str:
        with self._rec_lock:
            if self._writer is not None:
                try: self._writer.release()
                except Exception: pass
            self._writer = None
            self._recording = False
            path = self._record_path
            self._record_path = None
        return path or ""

    def is_recording(self) -> bool:
        with self._rec_lock:
            return self._recording

    def stop(self):
        self._running = False
        self.wait(2000)

# ---------- 파일 플레이어(overlay와 잘 겹치도록 QVideoSink 사용) ----------
class EmbeddedVideoPlayerLabel:
    """QMediaPlayer + QVideoSink로 videoRoad QLabel에 직접 렌더링"""
    def __init__(self, label: QLabel):
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
        self.label = label
        self._last_img = None
        self.player = QMediaPlayer(label)
        self.audio  = QAudioOutput(label)
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink(label)
        self.player.setVideoOutput(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)
        self.label.installEventFilter(_ResizeRelay(self._on_resize))

    def _on_frame(self, frame):
        try:
            img = frame.toImage()
        except Exception:
            return
        if img.isNull(): return
        self._last_img = img
        self._paint(img)

    def _paint(self, img: QImage):
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)

    def _on_resize(self):
        if self._last_img is not None:
            self._paint(self._last_img)

    def play(self, url: QUrl):
        self.player.setSource(url)
        self.player.play()

    def stop(self):
        self.player.stop()

# ---------- Custom LoadVideoDialog ----------
class LoadVideoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Video"); self.setModal(True)
        lay = QVBoxLayout(self)
        label = QLabel("Google Drive 링크(또는 파일 ID)나 로컬 파일 경로를 입력하세요.\n비워두면 파일 선택 창이 열립니다.")
        lay.addWidget(label)
        self.line = QLineEdit(self); lay.addWidget(self.line)
        row = QHBoxLayout(); btnCancel = QPushButton("Cancel", self); btnOk = QPushButton("OK", self)
        row.addStretch(1); row.addWidget(btnCancel); row.addWidget(btnOk); lay.addLayout(row)
        btnOk.clicked.connect(self.accept); btnCancel.clicked.connect(self.reject)
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
        self._recording = False
        self._record_dir = os.path.expanduser("~/Videos/bha_captures")
        os.makedirs(self._record_dir, exist_ok=True)

        # ===== 상단 로고(bhcLabel) =====
        self._init_bhc_logo()

        # ===== Mini dashboard (connectionPanel) =====
        self._setup_connection_dashboard()

        # ===== Log panel (logTable: 테이블만 표시, 버튼/주소 없음) =====
        self._setup_log_panel()

        # ===== Network stream → videoMain =====
        self._stream_worker = None
        self._last_qimg = None
        self._setup_stream_preview(STREAM_URL)

        # ===== videoRoad: QVideoSink 기반 플레이어 + Load 오버레이 =====
        self._player_target = None
        vr = getattr(self, "videoRoad", None)
        if isinstance(vr, QLabel):
            self._ensure_label_ready(vr)
            self._player_target = EmbeddedVideoPlayerLabel(vr)
            vr.setText("")
            self._overlay_road = self._build_overlay_on_label(vr, with_load=True, with_record=False, name="controlsOverlayRoad")

        # ---- 오버레이 컨트롤 바(Record/Stop) on videoMain ----
        vm = getattr(self, "videoMain", None)
        self._ensure_label_ready(vm)
        self._overlay_main = self._build_overlay_on_label(vm, with_load=False, with_record=True, name="controlsOverlayMain")

        # Heartbeat
        self._hb_stop = False
        threading.Thread(target=self._send_gui_heartbeat, daemon=True).start()

    # === 공통: 라벨 기본 스타일 ===
    def _ensure_label_ready(self, label: QLabel):
        if isinstance(label, QLabel):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background:#111; color:#ddd;")

    # === bhcLabel 세팅 ===
    def _init_bhc_logo(self):
        lbl = getattr(self, "bhcLabel", None)
        if not isinstance(lbl, QLabel): return
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background:#fff;")
        self._bhc_pix = QPixmap(BHC_IMG_PATH) if os.path.exists(BHC_IMG_PATH) else None
        if not self._bhc_pix:
            lbl.setText("bhc.png not found"); return
        def _repaint_logo():
            r = lbl.contentsRect()
            scaled = self._bhc_pix.scaled(r.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(scaled)
        self._bhc_relay = _ResizeRelay(_repaint_logo)
        lbl.installEventFilter(self._bhc_relay)
        _repaint_logo()

    # === videoMain/videoRoad 오버레이(버튼 바) 생성 ===
    def _build_overlay_on_label(self, label: QLabel, *, with_load: bool, with_record: bool, name: str):
        if not isinstance(label, QLabel): return None
        overlay = QWidget(label)
        overlay.setObjectName(name)
        overlay.setStyleSheet("background:rgba(255,255,255,0.75); border-radius:12px;")
        lay = QHBoxLayout(overlay); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(8)
        if with_load:
            btnLoad = QPushButton("Load", overlay)
            btnLoad.setStyleSheet("color: black; font-weight:600;")
            lay.addWidget(btnLoad)
            btnLoad.clicked.connect(self.handle_btnLoad_clicked)
        if with_record:
            self.btnRecord = QPushButton("⚫ Record", overlay)
            self.btnRecord.setStyleSheet("color:black; font-weight:600;")
            self.btnStopRec = QPushButton("■ Stop", overlay); self.btnStopRec.setEnabled(False)
            lay.addWidget(self.btnRecord); lay.addWidget(self.btnStopRec)
            self.btnRecord.clicked.connect(self._on_record_clicked)
            self.btnStopRec.clicked.connect(self._on_stop_record_clicked)
        def _pos():
            r = label.contentsRect()
            hint = overlay.sizeHint()
            ow, oh = hint.width(), hint.height()
            x = r.center().x() - ow // 2
            y = r.bottom() - oh - 8
            overlay.setGeometry(x, y, ow, oh)
            overlay.raise_(); overlay.show()
        _pos()
        label.installEventFilter(_ResizeRelay(_pos))
        return overlay

    # ----- Connection dashboard -----
    def _setup_connection_dashboard(self):
        panel = getattr(self, "connectionPanel", None)
        if not isinstance(panel, QWidget): return
        table = _ensure_conn_table_on(panel); _init_conn_table(table)
        _refresh_dashboard(table)
        self._conn_dash_timer = QTimer(self); self._conn_dash_timer.setInterval(2000)
        self._conn_dash_timer.timeout.connect(lambda: _refresh_dashboard(table)); self._conn_dash_timer.start()
        btn = getattr(self, "btnReconnect", None)
        if isinstance(btn, QPushButton): btn.clicked.connect(lambda: _refresh_dashboard(table))

    # ----- Log panel (logTable 내부: 테이블만) -----
    def _setup_log_panel(self):
        container = getattr(self, "logTable", None)
        if not isinstance(container, QWidget): return
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # 헤더/버튼/주소 없이 QTableWidget만 배치
        self._logTable = QTableWidget(0, 3, container)
        self._logTable.setObjectName("vqaTable")
        self._logTable.setHorizontalHeaderLabels(["question", "answer", "created_at"])
        self._logTable.verticalHeader().setVisible(False)
        hdr = self._logTable.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self._logTable, 1)

        # 자동 새로고침(15s)만 유지, UI에 버튼/주소 표시 없음
        self._logTimer = QTimer(self); self._logTimer.setInterval(15000)
        self._logTimer.timeout.connect(self._refresh_vqa_log)
        self._logTimer.start()

        # 최초 1회 로드
        QTimer.singleShot(0, self._refresh_vqa_log)

    def _refresh_vqa_log(self):
        limit = 500  # 고정 (UI 요소 없음)
        rows = []
        try:
            import pymysql
            conn = pymysql.connect(
                host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
                database=DB_NAME, charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor, autocommit=True
            )
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT question, answer, created_at FROM vqa_log ORDER BY id DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
        except Exception:
            traceback.print_exc()
            return

        t = self._logTable; t.setRowCount(0)
        for r in rows:
            row = t.rowCount(); t.insertRow(row)
            t.setItem(row, 0, QTableWidgetItem(str(r.get("question",""))))
            t.setItem(row, 1, QTableWidgetItem(str(r.get("answer",""))))
            t.setItem(row, 2, QTableWidgetItem(str(r.get("created_at",""))))

    # ----- GUI Heartbeat sender -----
    def _send_gui_heartbeat(self):
        try: ip = socket.gethostbyname(socket.gethostname())
        except Exception: ip = "127.0.0.1"
        while not self._hb_stop:
            try:
                import pymysql
                conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME, autocommit=True)
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO system_heartbeat (component, ip, status, last_seen)
                        VALUES ('BHC_GUI', %s, 'OK', NOW())
                        ON DUPLICATE KEY UPDATE status=VALUES(status), last_seen=NOW()
                    """, (ip,))
            except Exception:
                pass
            time.sleep(5)

    # ---------- Network stream → videoMain ----------
    def _setup_stream_preview(self, url: str):
        target = getattr(self, "videoMain", None)
        if not isinstance(target, QLabel): return
        self._ensure_label_ready(target)
        target.setText("Streaming 준비 중…")
        self._relay_vm = _ResizeRelay(self._resize_videoMain)
        target.installEventFilter(self._relay_vm)
        self._stream_worker = NetworkStreamWorker(url, parent=self)
        self._stream_worker.frameReady.connect(self._on_stream_frame)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_worker.start()
        self.statusBar.showMessage(f"Streaming: {url} → videoMain")

    def _on_stream_frame(self, qimg: QImage):
        self._last_qimg = qimg
        self._paint_videoMain(qimg)

    def _on_stream_error(self, msg: str):
        self.statusBar.showMessage(msg)
        QMessageBox.warning(self, "Stream", msg)
        if self._recording and self._stream_worker is not None:
            try:
                self._stream_worker.stop_recording()
                self._recording = False
                if hasattr(self, "btnRecord"): self.btnRecord.setEnabled(True)
                if hasattr(self, "btnStopRec"): self.btnStopRec.setEnabled(False)
                if hasattr(self, "btnRecord"): self.btnRecord.setText("⚫ Record")
            except Exception:
                pass

    # ---------- Recording (Stream only) ----------
    def _on_record_clicked(self):
        if self._stream_worker is None:
            QMessageBox.warning(self, "Record", "스트림이 시작되지 않았습니다.")
            return
        if self._recording: return
        filename = f"record_{now_ts()}.mp4"
        path = os.path.join(self._record_dir, filename)
        ok = self._stream_worker.start_recording(path)
        if not ok:
            QMessageBox.warning(self, "Record", "녹화 시작 실패(코덱/권한 확인).")
            return
        self._recording = True
        if hasattr(self, "btnRecord"): self.btnRecord.setEnabled(False)
        if hasattr(self, "btnStopRec"): self.btnStopRec.setEnabled(True)
        if hasattr(self, "btnRecord"): self.btnRecord.setText("🔴 Record")
        self.statusBar.showMessage(f"Recording (Stream)... -> {path}")

    def _on_stop_record_clicked(self):
        if not self._recording: return
        saved = ""
        if self._stream_worker is not None:
            try: saved = self._stream_worker.stop_recording()
            except Exception as e:
                traceback.print_exc(); QMessageBox.critical(self, "Record", f"녹화 정지 실패: {e}")
        self._recording = False
        if hasattr(self, "btnRecord"): self.btnRecord.setEnabled(True)
        if hasattr(self, "btnStopRec"): self.btnStopRec.setEnabled(False)
        if hasattr(self, "btnRecord"): self.btnRecord.setText("⚫ Record")
        msg = f"Saved: {saved}" if saved else "Saved (path unknown)"
        drive_msg = try_save_to_drive(saved) if saved else "경로 미상"
        self.statusBar.showMessage(f"{msg} | {drive_msg}")
        QMessageBox.information(self, "Record", f"{msg}\n{drive_msg}")

    # ---------- Load(파일/드라이브) → videoRoad ----------
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
                target = getattr(self, "videoRoad", None)
                if isinstance(target, QLabel):
                    self._player_target = EmbeddedVideoPlayerLabel(target)
            if self._player_target:
                self._player_target.play(url)
                self.statusBar.showMessage(f"Playing: {url.toString()}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Play Error", f"영상을 재생하지 못했습니다:\n{e}")
        finally:
            self._loadDialogOpen = False

    # ---------- Painting ----------
    def _paint_videoMain(self, qimg: QImage):
        target = getattr(self, "videoMain", None)
        if not isinstance(target, QLabel): return
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        target.setPixmap(scaled)

    def _resize_videoMain(self):
        if hasattr(self, "_last_qimg") and self._last_qimg is not None:
            self._paint_videoMain(self._last_qimg)

    # ---------- Graceful stop ----------
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(e)

    def closeEvent(self, event):
        self._hb_stop = True
        try: self._conn_dash_timer.stop()
        except Exception: pass
        try:
            if hasattr(self, "_logTimer") and self._logTimer is not None:
                self._logTimer.stop()
        except Exception: pass
        try:
            if self._recording and self._stream_worker is not None:
                self._stream_worker.stop_recording()
                self._recording = False
        except Exception: pass
        try:
            if self._stream_worker is not None:
                self._stream_worker.stop()
        except Exception: pass
        super().closeEvent(event)

# ---------- main ----------
if __name__ == '__main__':
    try: import pymysql
    except Exception: print("TIP: pip install pymysql")
    try:
        from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
        from PyQt6.QtMultimediaWidgets import QVideoWidget
    except Exception:
        print("TIP: pip install PyQt6 PyQt6-Qt6 (and gstreamer plugins)")

    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
