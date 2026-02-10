import sys
import os
import hashlib
import numpy as np
import pygame
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, QTimer
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

# -----------------------------
# PAPER-DEFINED CONSTANTS
# -----------------------------
# Per paper: silence = any NON-VOCAL interval >= 2.0 seconds (long pause)
SILENCE_MIN_SEC = 2.0

# Per paper: boundary editing at ~50 ms precision
BOUNDARY_ZOOM_HALF_WINDOW_SEC = 0.05  # +/- 50 ms

# Safe merging of extremely close intervals (kept; does NOT define silence)
MERGE_GAP_SEC = 0.05  # 50 ms

# Sliding window for energy scanning (ms)
SCAN_STEP_MS = 10


def calculate_dynamic_noise_floor(audio_segment: AudioSegment, sample_duration=100, num_samples=30) -> float:
    """
    Estimate a dynamic noise floor by sampling dBFS across the file.
    Returns mean dBFS over sampled windows.
    """
    segment_length = len(audio_segment)
    if segment_length == 0:
        return -60.0  # fallback

    step = max(segment_length // num_samples, sample_duration)
    noise_levels = []

    for i in range(0, segment_length, step):
        start = i
        end = min(i + sample_duration, segment_length)
        seg = audio_segment[start:end]
        noise_levels.append(seg.dBFS)

    return float(np.mean(noise_levels))


def adaptive_silence_detection(audio_segment: AudioSegment, min_silence_len_ms: int, noise_padding_db: float):
    """
    Detect candidate non-vocal intervals using adaptive threshold:
    silence_thresh = noise_floor + noise_padding_db.

    Returns intervals in milliseconds: [(start_ms, end_ms), ...]
    """
    noise_floor = calculate_dynamic_noise_floor(audio_segment)
    silence_thresh = noise_floor + noise_padding_db

    intervals_ms = []
    start = None

    for i in range(0, len(audio_segment), SCAN_STEP_MS):
        seg = audio_segment[i:i + SCAN_STEP_MS]
        if seg.dBFS < silence_thresh:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_silence_len_ms:
                    intervals_ms.append((start, i))
                start = None

    # trailing silence
    if start is not None and len(audio_segment) - start >= min_silence_len_ms:
        intervals_ms.append((start, len(audio_segment)))

    return intervals_ms, silence_thresh, noise_floor


def merge_intervals(intervals_sec, merge_gap=MERGE_GAP_SEC):
    """
    Merge overlapping or very close intervals.
    intervals_sec: list[(start_sec, end_sec)]
    """
    if not intervals_sec:
        return []

    intervals_sec = sorted(intervals_sec, key=lambda x: x[0])
    merged = [intervals_sec[0]]

    for start, end in intervals_sec[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + merge_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class AudioProcessor(QMainWindow):
    """
    Paper-matching pause tool:
    - Only intervals >= SILENCE_MIN_SEC are kept & saved.
    - Saved metric: SilenceToAudioRatio = total_silence_time(audio) / audio_length.
    - Provides 50 ms zoom around start/end boundary of last interval.
    """

    def __init__(self):
        super().__init__()
        self.audio: AudioSegment | None = None
        self.audio_file_path: str | None = None

        # stored SILENCE intervals (paper definition): list of (start_sec, end_sec), each >= 2.0s
        self.silence_intervals = []

        self.span_selector = None
        self.current_meta = {}
        self.current_file_number = None

        # playback state
        self.is_paused = False
        self.play_thread = None
        self.playback_pos_sec = 0.0
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self.update_playback_cursor)
        self.cursor_line = None

        # for saving
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.excel_path = os.path.join(self.output_dir, "silence_metrics.xlsx")

        # store last detection params for audit trail
        self.last_noise_floor = None
        self.last_silence_thresh = None
        self.last_noise_padding_db = None

        pygame.mixer.init()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Silence/Pause Annotation (Paper-Matched)')
        self.setGeometry(100, 100, 1100, 850)

        layout = QVBoxLayout()

        # status label
        self.statusLabel = QLabel("Load a WAV file to begin.")
        layout.addWidget(self.statusLabel)

        # Matplotlib canvas
        self.canvas = FigureCanvas(plt.Figure(figsize=(12, 6)))
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        # Buttons row
        btn_row = QHBoxLayout()

        self.openButton = QPushButton('Open WAV')
        self.openButton.clicked.connect(self.open_file)
        btn_row.addWidget(self.openButton)

        self.processButton = QPushButton('Auto-detect (>=2.0s candidates)')
        self.processButton.clicked.connect(self.process_audio)
        btn_row.addWidget(self.processButton)

        self.playButton = QPushButton('Play')
        self.playButton.clicked.connect(self.play_audio)
        btn_row.addWidget(self.playButton)

        self.pauseButton = QPushButton('Pause/Resume')
        self.pauseButton.clicked.connect(self.pause_resume_audio)
        btn_row.addWidget(self.pauseButton)

        self.saveButton = QPushButton('Save Metrics (SilenceToAudioRatio)')
        self.saveButton.clicked.connect(self.calculate_and_save_metrics)
        btn_row.addWidget(self.saveButton)

        layout.addLayout(btn_row)

        # Noise padding slider (kept; paper says candidates were verified/edited)
        self.noisePadLabel = QLabel('Noise Padding (dB) for candidate detection:')
        layout.addWidget(self.noisePadLabel)

        self.noisePadSlider = QSlider(Qt.Horizontal)
        self.noisePadSlider.setMinimum(0)
        self.noisePadSlider.setMaximum(20)
        self.noisePadSlider.setValue(5)
        layout.addWidget(self.noisePadSlider)

        # Min silence length slider (paper-fixed at 2s) -> lock & disable
        self.minSilenceLabel = QLabel(f'Minimum Silence Length: {int(SILENCE_MIN_SEC*1000)} ms (locked to match paper)')
        layout.addWidget(self.minSilenceLabel)

        self.minSilenceSlider = QSlider(Qt.Horizontal)
        self.minSilenceSlider.setMinimum(int(SILENCE_MIN_SEC * 1000))
        self.minSilenceSlider.setMaximum(int(SILENCE_MIN_SEC * 1000))
        self.minSilenceSlider.setValue(int(SILENCE_MIN_SEC * 1000))
        self.minSilenceSlider.setEnabled(False)
        layout.addWidget(self.minSilenceSlider)

        # Save location hint
        self.savePathLabel = QLabel(f"Saving to: {self.excel_path}")
        layout.addWidget(self.savePathLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        # Keyboard shortcuts on the canvas
        # z = undo last interval
        # c = clear all intervals
        # 1 = zoom to start boundary of last interval (±50ms)
        # 2 = zoom to end boundary of last interval (±50ms)
        # 0 = reset zoom to full file
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    # ----------------- File loading & plotting -----------------

    def open_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "WAV Files (*.wav);;All Files (*)",
            options=options
        )
        self.plot_waveform()
        self.canvas.setFocus()
        if not fileName:
            return

        self.audio_file_path = fileName
        self.audio = AudioSegment.from_wav(self.audio_file_path)

        # Convert to mono for consistent dBFS + plotting (optional but recommended)
        if self.audio.channels > 1:
            self.audio = self.audio.set_channels(1)

        self.current_file_number = os.path.basename(self.audio_file_path).split('.')[0]

        # reset cursor & intervals
        self.playback_pos_sec = 0.0
        self.silence_intervals = []
        self.cursor_line = None

        self.extract_meta(fileName)
        self.plot_waveform()
        self.status("Loaded file. Use auto-detect or manual selection; only >=2.0s will be kept.")

    def plot_waveform(self):
        if self.audio is None:
            return

        samples = np.array(self.audio.get_array_of_samples())
        sr = self.audio.frame_rate
        times = np.arange(len(samples)) / sr

        self.ax.clear()
        self.ax.plot(times, samples, lw=0.5)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Waveform')

        # draw cursor
        self.cursor_line = self.ax.axvline(self.playback_pos_sec, color='green', linewidth=1)
        self.canvas.draw()
        self.init_span_selector()

    def init_span_selector(self):
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
            self.span_selector = None

        self.span_selector = SpanSelector(
            self.ax,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True
        )

    def update_waveform(self):
        if self.audio is None:
            return

        samples = np.array(self.audio.get_array_of_samples())
        sr = self.audio.frame_rate
        times = np.arange(len(samples)) / sr

        self.ax.clear()
        self.ax.plot(times, samples, lw=0.5)

        # draw silence regions
        for start, stop in self.silence_intervals:
            self.ax.axvspan(start, stop, color='red', alpha=0.5)

        # cursor
        self.cursor_line = self.ax.axvline(self.playback_pos_sec, color='green', linewidth=1)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title(f'Waveform with SILENCE intervals (>= {SILENCE_MIN_SEC:.1f}s)')
        self.canvas.draw()
        self.init_span_selector()
        self.canvas.draw()
        self.canvas.setFocus()

    # ----------------- Selection callbacks -----------------

    def on_select(self, start, stop):
        if stop < start:
            start, stop = stop, start

        dur = stop - start
        if dur < SILENCE_MIN_SEC:
            self.status(f"Ignored interval {dur:.2f}s (< {SILENCE_MIN_SEC:.1f}s). Paper definition requires >=2.0s.")
            return

        self.silence_intervals.append((start, stop))
        self.silence_intervals = merge_intervals(self.silence_intervals, merge_gap=MERGE_GAP_SEC)

        self.update_waveform()
        self.status(f"Added silence interval {dur:.2f}s (>=2.0s). Press 1/2 to zoom boundaries ±50ms.")
        self.canvas.setFocus()


    def on_key_press(self, event):
        print("KEY EVENT:", event.key)
        if event.key == "z":
            if self.silence_intervals:
                removed = self.silence_intervals.pop()
                self.status(f"Undo last interval: {removed}")
                self.update_waveform()
            else:
                self.status("No intervals to undo.")

        elif event.key == "c":
            self.silence_intervals = []
            self.status("Cleared all intervals.")
            self.update_waveform()

        elif event.key == "1":
            self.zoom_last_boundary(which="start")

        elif event.key == "2":
            self.zoom_last_boundary(which="end")

        elif event.key == "0":
            self.reset_zoom()

    def zoom_last_boundary(self, which: str):
        if self.audio is None or not self.silence_intervals:
            self.status("No intervals to zoom.")
            return

        start, end = self.silence_intervals[-1]
        boundary = start if which == "start" else end

        audio_len = len(self.audio) / 1000.0
        left = clamp(boundary - BOUNDARY_ZOOM_HALF_WINDOW_SEC, 0.0, audio_len)
        right = clamp(boundary + BOUNDARY_ZOOM_HALF_WINDOW_SEC, 0.0, audio_len)

        # ensure visible even at file edges
        if right - left < 2 * BOUNDARY_ZOOM_HALF_WINDOW_SEC:
            right = clamp(left + 2 * BOUNDARY_ZOOM_HALF_WINDOW_SEC, 0.0, audio_len)

        self.ax.set_xlim(left, right)
        self.canvas.draw_idle()
        self.status(f"Zoomed to {which} boundary at {boundary:.3f}s (±50ms). Adjust by dragging selection edges.")

    def reset_zoom(self):
        if self.audio is None:
            return
        audio_len = len(self.audio) / 1000.0
        self.ax.set_xlim(0, audio_len)
        self.canvas.draw_idle()
        self.status("Reset zoom to full file view.")

    # ----------------- Processing -----------------

    def process_audio(self):
        if self.audio is None:
            self.status("No audio loaded.")
            return

        noise_padding_db = float(self.noisePadSlider.value())
        min_silence_len_ms = int(SILENCE_MIN_SEC * 1000)  # locked to paper

        intervals_ms, silence_thresh, noise_floor = adaptive_silence_detection(
            self.audio,
            min_silence_len_ms=min_silence_len_ms,
            noise_padding_db=noise_padding_db
        )

        self.last_noise_floor = noise_floor
        self.last_silence_thresh = silence_thresh
        self.last_noise_padding_db = noise_padding_db

        # Convert to seconds
        cand = [(s / 1000.0, e / 1000.0) for s, e in intervals_ms]

        # Enforce paper definition again (belt & suspenders)
        cand = [(s, e) for s, e in cand if (e - s) >= SILENCE_MIN_SEC]

        self.silence_intervals = merge_intervals(cand, merge_gap=MERGE_GAP_SEC)
        self.update_waveform()

        self.status(
            f"Auto-detected {len(self.silence_intervals)} candidate silences (>=2.0s). "
            f"NoiseFloor={noise_floor:.1f} dBFS, Thresh={silence_thresh:.1f} dBFS. "
            "Visually verify/edit each interval. (Filled pauses should be excluded manually.)"
        )

    def extract_meta(self, file_path: str):
        """
        Expected optional folder structure:
        .../<group>/<gender>/<participant_dir>/<file>.wav

        IMPORTANT: We DO NOT store participant names.
        We store a stable hash of participant_dir instead.
        """
        path_parts = file_path.replace('\\', '/').split('/')
        meta = {}

        # best-effort extraction without crashing
        try:
            group = path_parts[-5]
            gender = path_parts[-4]
            participant_dir = path_parts[-3]
            pid = hashlib.sha1(participant_dir.encode("utf-8")).hexdigest()[:12]
            meta.update({"group": group, "gender": gender, "participant_id": pid})
        except Exception:
            meta.update({"group": "", "gender": "", "participant_id": ""})

        self.current_meta = meta
        return True

    # ----------------- Playback & cursor -----------------

    def play_audio(self):
        if self.audio is None or self.audio_file_path is None:
            self.status("No audio loaded.")
            return

        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.playback_timer.start()
            self.status("Resumed.")
            return

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        self.playback_pos_sec = 0.0
        if self.cursor_line is not None:
            self.cursor_line.set_xdata([self.playback_pos_sec, self.playback_pos_sec])
            self.canvas.draw_idle()

        self.play_thread = PlayAudioThread(self.audio_file_path)
        self.play_thread.start()
        self.playback_timer.start()
        self.is_paused = False
        self.status("Playing...")

    def pause_resume_audio(self):
        if self.audio is None or self.audio_file_path is None:
            self.status("No audio loaded.")
            return

        if not self.is_paused and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.is_paused = True
            self.playback_timer.stop()
            self.status("Paused.")
        elif self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.playback_timer.start()
            self.status("Resumed.")
        else:
            self.status("No audio playing.")

    def update_playback_cursor(self):
        if self.audio is None or self.cursor_line is None:
            return

        if not pygame.mixer.music.get_busy() and not self.is_paused:
            self.playback_timer.stop()
            self.playback_pos_sec = len(self.audio) / 1000.0
            self.cursor_line.set_xdata([self.playback_pos_sec, self.playback_pos_sec])
            self.canvas.draw_idle()
            return

        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:
            pos_ms = 0

        self.playback_pos_sec = pos_ms / 1000.0
        self.cursor_line.set_xdata([self.playback_pos_sec, self.playback_pos_sec])
        self.canvas.draw_idle()

    # ----------------- Metrics & saving -----------------

    def calculate_and_save_metrics(self):
        if self.audio is None:
            self.status("No audio loaded.")
            return

        # Merge & enforce paper definition
        merged = merge_intervals(self.silence_intervals, merge_gap=MERGE_GAP_SEC)
        merged = [(s, e) for s, e in merged if (e - s) >= SILENCE_MIN_SEC]
        self.silence_intervals = merged

        audio_length_sec = len(self.audio) / 1000.0
        total_silence_sec = sum(e - s for s, e in merged)
        num_pauses = len(merged)
        mean_pause_sec = (total_silence_sec / num_pauses) if num_pauses else 0.0

        silence_to_audio_ratio = (total_silence_sec / audio_length_sec) if audio_length_sec else 0.0

        self.update_waveform()

        self.status(
            f"Saved: total_silence={total_silence_sec:.2f}s, pauses={num_pauses}, "
            f"SilenceToAudioRatio={silence_to_audio_ratio:.3f}"
        )

        self.save_to_excel(
            meta=self.current_meta,
            file_number=self.current_file_number,
            audio_length_sec=audio_length_sec,
            num_pauses=num_pauses,
            mean_pause_sec=mean_pause_sec,
            total_silence_sec=total_silence_sec,
            silence_to_audio_ratio=silence_to_audio_ratio
        )

    def save_to_excel(
        self,
        meta,
        file_number,
        audio_length_sec,
        num_pauses,
        mean_pause_sec,
        total_silence_sec,
        silence_to_audio_ratio
    ):
        # Ensure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)

        cols = [
            "ParticipantID", "Gender", "Group", "FileNumber",
            "AudioLengthSec",
            "NumSilencePauses_ge2s",
            "MeanSilencePauseSec_ge2s",
            "TotalSilenceSec_ge2s",
            "SilenceToAudioRatio",
            "SilenceMinSec",
            "MergeGapSec",
            "NoisePaddingDB",
            "NoiseFloorDBFS",
            "SilenceThreshDBFS",
        ]

        if os.path.exists(self.excel_path):
            df = pd.read_excel(self.excel_path)
        else:
            df = pd.DataFrame(columns=cols)

        new_row = {
            "ParticipantID": meta.get("participant_id", ""),
            "Gender": meta.get("gender", ""),
            "Group": meta.get("group", ""),
            "FileNumber": file_number,
            "AudioLengthSec": float(audio_length_sec),
            "NumSilencePauses_ge2s": int(num_pauses),
            "MeanSilencePauseSec_ge2s": float(mean_pause_sec),
            "TotalSilenceSec_ge2s": float(total_silence_sec),
            "SilenceToAudioRatio": float(silence_to_audio_ratio),
            "SilenceMinSec": float(SILENCE_MIN_SEC),
            "MergeGapSec": float(MERGE_GAP_SEC),
            "NoisePaddingDB": float(self.last_noise_padding_db) if self.last_noise_padding_db is not None else np.nan,
            "NoiseFloorDBFS": float(self.last_noise_floor) if self.last_noise_floor is not None else np.nan,
            "SilenceThreshDBFS": float(self.last_silence_thresh) if self.last_silence_thresh is not None else np.nan,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(self.excel_path, index=False)

        self.savePathLabel.setText(f"Saving to: {self.excel_path}")
        print(f"[OK] Saved to {self.excel_path}")

    # ----------------- helpers -----------------

    def status(self, msg: str):
        self.statusLabel.setText(msg)
        print(msg)


class PlayAudioThread(QThread):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            pygame.mixer.music.load(self.file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Playback error: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioProcessor()
    ex.show()
    sys.exit(app.exec_())
