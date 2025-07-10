import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import pyfftw

fft_module = pyfftw.interfaces.numpy_fft
import pyaudio
import threading
import time
import os
import glob
from pydub import AudioSegment
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from typing import List, Dict, Any, Optional
import sys
import platform

CHUNK_SIZE = 1024 * 4
MAX_GAIN_DB = 12.0
SEEK_SECONDS = 5
FREQ_BANDS = [60, 170, 310, 600, 1000, 3000, 6000, 12000, 14000, 16000]
BAND_LABELS = [
    "Sub-Bass",
    "Betonung",
    "Wärme",
    "Klarheit",
    "Vokal",
    "Lebendigkeit",
    "Helligkeit",
    "Glanz",
    "Luft",
    "Ultra-Höhen",
]

NUM_VIS_BARS = 30
VIS_MIN_DB = -60.0
VIS_MAX_DB = 15.0
VIS_DB_RANGE = VIS_MAX_DB - VIS_MIN_DB

MUSIC_CATALOG: List[Dict[str, Any]] = [
    {
        "composer": "Ludwig van Beethoven",
        "title": "Sinfonie Nr. 5 c-Moll, Op. 67",
        "description": """Die 5. Sinfonie von Beethoven, komponiert zwischen 1804 und 1808, zählt zu den berühmtesten Werken der klassischen Musik. Das markante Anfangsmotiv – oft als „Schicksalsmotiv“ bezeichnet – ist weltbekannt und steht für den dramatischen Charakter des gesamten Werkes. Die Uraufführung fand am 22. Dezember 1808 im Theater an der Wien statt und wurde bereits zu Beethovens Lebzeiten ein großer Erfolg.""",
        "filename": "Beethoven Symphony Sinfonie 5.mp3",
        "filepath": None,
    },
    {
        "composer": "Franz Schubert",
        "title": "Sinfonie Nr. 5 in B-Dur, D 485",
        "description": """Die 5. Sinfonie von Franz Schubert wurde 1816 komponiert, als der Komponist erst 19 Jahre alt war. Sie zeichnet sich durch ihren leichten, heiteren Charakter aus und erinnert stilistisch an Mozart. Zu Schuberts Lebzeiten wurde das Werk nie öffentlich aufgeführt, sondern erst am 17. Oktober 1841 in Wien uraufgeführt.""",
        "filename": "Schubert Symphony Sinfonie 5.mp3",
        "filepath": None,
    },
    {
        "composer": "Wolfgang Amadeus Mozart",
        "title": "Ouvertüre zu „Le Nozze di Figaro“, KV 492",
        "description": """Die Ouvertüre zu Mozarts Oper „Le Nozze di Figaro“ entstand 1786 und gilt als eines der spritzigsten Orchesterstücke der Klassik. Sie ist lebhaft, voller Energie und bereitet das Publikum auf die heitere Oper vor. Die Uraufführung fand zusammen mit der Oper am 1. Mai 1786 im Burgtheater in Wien statt.""",
        "filename": "Mozart Figaro Ouvertüre.mp3",
        "filepath": None,
    },
    {
        "composer": "Joseph Haydn",
        "title": "Sinfonie Nr. 94 G-Dur, „Mit dem Paukenschlag“",
        "description": """Die 94. Sinfonie von Joseph Haydn wurde 1791 in London komponiert. Ihren Beinamen „Mit dem Paukenschlag“ verdankt sie dem überraschend lauten Paukenschlag im zweiten Satz, der das Publikum aufschrecken sollte. Die Uraufführung am 23. März 1792 in London war ein großer Erfolg.""",
        "filename": "Haydn Symphony Sinfonie 94 Paukenschlag.mp3",
        "filepath": None,
    },
    {
        "composer": "Ludwig van Beethoven",
        "title": "Sinfonie Nr. 8 F-Dur, Op. 93",
        "description": """Beethovens 8. Sinfonie wurde 1812 vollendet und hebt sich durch ihren humorvollen und leichten Charakter ab. Beethoven selbst nannte sie gerne seine „kleine Sinfonie in F“. Obwohl sie zunächst im Schatten der 7. Sinfonie stand, gilt sie heute als Meisterwerk voller Witz und Lebendigkeit.""",
        "filename": "Beethoven Symphony Sinfonie 8.mp3",
        "filepath": None,
    },
    {
        "composer": "Ludwig van Beethoven",
        "title": "Klaviersonate Nr. 17 d-Moll, Op. 31 Nr. 2 („Sturm-Sonate“)",
        "description": """Beethovens 17. Klaviersonate, komponiert 1802, ist auch als „Sturm-Sonate“ bekannt. Das Werk beeindruckt durch seine dramatische Tiefe, innovative Harmonik und leidenschaftlichen Ausdruck. Der dritte Satz, Allegretto, ist besonders virtuos und voller Energie, was den Beinamen der Sonate eindrucksvoll widerspiegelt.""",
        "filename": "Beethoven Piano Sonata 17 Sturm Tempest.mp3",
        "filepath": None,
    },
]


def resource_path(relative_path):
    """Hem script modunda hem de PyInstaller ile paketlenmiş halde
    varlıkların doğru yolunu döndürür."""
    try:
        # PyInstaller geçici bir klasör oluşturur ve yolu _MEIPASS içinde saklar
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# =============================================================================
# --- Pydub için FFmpeg yolunu ayarla ---
if platform.system() == "Windows":
    AudioSegment.converter = resource_path("ffmpeg/windows/ffmpeg")
elif platform.system() == "Darwin":  # macOS
    AudioSegment.converter = resource_path("ffmpeg/macos/ffmpeg")
else:
    AudioSegment.converter = resource_path("ffmpeg/ffmpeg")


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(seconds, 60)
    return f"{int(mins):02d}:{int(secs):02d}"


def design_peaking_filter(center_freq, q_factor, gain_db, fs):
    """Peaking EQ filtresi tasarlar ve katsayıları döndürür"""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * center_freq / fs
    q_factor = max(q_factor, 0.01)
    alpha = np.sin(w0) / (2 * q_factor)

    # Biquad filter coefficients
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Normalize coefficients
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return {"b": b, "a": a}


# --- Numba ile hızlandırılmış zincirleme filtre uygulaması ---
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def chain_biquad_filters(data, filters_b, filters_a, filters_zi, num_filters):
        out = data.copy()
        for f in range(num_filters):
            b = filters_b[f]
            a = filters_a[f]
            z = filters_zi[f]
            z0, z1 = z[0], z[1]
            for n in range(len(out)):
                w = out[n] - a[1] * z0 - a[2] * z1
                out[n] = b[0] * w + b[1] * z0 + b[2] * z1
                z1, z0 = z0, w
            filters_zi[f, 0], filters_zi[f, 1] = z0, z1
        return out, filters_zi

else:

    def chain_biquad_filters(data, filters_b, filters_a, filters_zi, num_filters):
        out = data.copy()
        for f in range(num_filters):
            b = filters_b[f]
            a = filters_a[f]
            z = filters_zi[f]
            z0, z1 = z[0], z[1]
            for n in range(len(out)):
                w = out[n] - a[1] * z0 - a[2] * z1
                out[n] = b[0] * w + b[1] * z0 + b[2] * z1
                z1, z0 = z0, w
            filters_zi[f, 0], filters_zi[f, 1] = z0, z1
        return out, filters_zi


class AudioEngine:
    def __init__(
        self, on_progress_update, on_visualization_update, on_playback_stopped
    ):
        self.p = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.song: Optional[AudioSegment] = None
        self.audio_data: Optional[np.ndarray] = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.playback_thread: Optional[threading.Thread] = None
        self.playback_lock = threading.Lock()

        self.filters: List[Optional[Dict]] = [None] * len(FREQ_BANDS)
        self.filter_states: List[Optional[np.ndarray]] = [None] * len(FREQ_BANDS)
        self.ui_progress_callback = on_progress_update
        self.ui_visualizer_callback = on_visualization_update
        self.ui_stop_callback = on_playback_stopped

    def update_eq(self, gains_db: List[float], q_factor: float):
        """
        Optimize: Sadece değişen filtrelerin katsayılarını ve state'lerini güncelle.
        Aynı kalan filtrelerin state'ini koru, gereksiz yere sıfırlama yapma.
        """
        if not self.song:
            return
        with self.playback_lock:
            for i, gain in enumerate(gains_db):
                prev_filter = self.filters[i]
                prev_gain = None
                prev_q = None
                if prev_filter is not None:
                    # Eski gain ve q değerlerini çözümle (katsayıdan çıkarılamaz, ama optimize için not)
                    pass
                if abs(gain) < 0.1:
                    # Kapalıysa filtreyi ve state'i kaldır
                    if self.filters[i] is not None or self.filter_states[i] is not None:
                        self.filters[i], self.filter_states[i] = None, None
                else:
                    # Yeni katsayıyı oluştur
                    new_filter = design_peaking_filter(
                        FREQ_BANDS[i], q_factor, gain, self.song.frame_rate
                    )
                    # Eğer filtre katsayıları değişmediyse state'i koru
                    if (
                        self.filters[i] is not None
                        and np.allclose(
                            self.filters[i]["b"], new_filter["b"], atol=1e-6
                        )
                        and np.allclose(
                            self.filters[i]["a"], new_filter["a"], atol=1e-6
                        )
                    ):
                        # Aynı filtre, state'i koru
                        pass
                    else:
                        # Farklıysa state'i sıfırla
                        self.filter_states[i] = np.zeros((self.song.channels, 2))
                    self.filters[i] = new_filter

    def _playback_loop(self):
        if not self.song or self.audio_data is None:
            return
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.song.channels,
            rate=self.song.frame_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        while self.is_playing:
            while self.is_paused and self.is_playing:
                time.sleep(0.1)
            if not self.is_playing:
                break
            with self.playback_lock:
                start, end = self.current_frame, self.current_frame + CHUNK_SIZE
                if end > len(self.audio_data):
                    break
                chunk, self.current_frame = self.audio_data[start:end], end
            processed_chunk = self._apply_filters(chunk)
            self.stream.write(processed_chunk.tobytes())
            self.ui_progress_callback(self.current_frame, self.song.duration_seconds)
            self.ui_visualizer_callback(
                processed_chunk[:, 0] if processed_chunk.ndim == 2 else processed_chunk
            )
        self.stop()
        self.ui_stop_callback()

    def _apply_filters(self, chunk: np.ndarray) -> np.ndarray:
        """
        Optimize: Apply all active filters in a single loop (Numba/NumPy),
        process both channels together if stereo, minimize Python overhead.
        """
        processed_chunk = chunk.copy()
        with self.playback_lock:
            # Aktif filtrelerin indekslerini topla
            active_idx = [
                i
                for i, f in enumerate(self.filters)
                if f is not None and self.filter_states[i] is not None
            ]
            if not active_idx:
                return np.clip(processed_chunk, -1.0, 1.0)

            num_filters = len(active_idx)
            # Mono
            if processed_chunk.ndim == 1:
                filters_b = np.stack([self.filters[i]["b"] for i in active_idx])
                filters_a = np.stack([self.filters[i]["a"] for i in active_idx])
                filters_zi = np.stack([self.filter_states[i][0] for i in active_idx])
                out, new_zi = chain_biquad_filters(
                    processed_chunk, filters_b, filters_a, filters_zi, num_filters
                )
                processed_chunk = out
                for idx, i in enumerate(active_idx):
                    self.filter_states[i][0] = new_zi[idx]
            # Stereo
            elif processed_chunk.ndim == 2 and processed_chunk.shape[1] == 2:
                for ch in range(2):
                    filters_b = np.stack([self.filters[i]["b"] for i in active_idx])
                    filters_a = np.stack([self.filters[i]["a"] for i in active_idx])
                    filters_zi = np.stack(
                        [self.filter_states[i][ch] for i in active_idx]
                    )
                    out, new_zi = chain_biquad_filters(
                        processed_chunk[:, ch],
                        filters_b,
                        filters_a,
                        filters_zi,
                        num_filters,
                    )
                    processed_chunk[:, ch] = out
                    for idx, i in enumerate(active_idx):
                        self.filter_states[i][ch] = new_zi[idx]
            else:
                # Unexpected shape, fallback
                pass
        return np.clip(processed_chunk, -1.0, 1.0)

    def load_song(self, filepath: str) -> bool:
        try:
            self.stop()
            self.song = AudioSegment.from_mp3(filepath)
            self.audio_data = np.array(self.song.get_array_of_samples()).reshape(
                -1, self.song.channels
            )
            self.audio_data = self.audio_data.astype(np.float32) / (
                2 ** (self.song.sample_width * 8 - 1)
            )
            self.current_frame, self.is_paused = 0, False
            self._initialize_filter_states()
            return True
        except Exception as e:
            messagebox.showerror(
                "Ladefehler", f"Datei konnte nicht geladen werden:\n{e}"
            )
            return False

    def play(self):
        if not self.is_playing and self.audio_data is not None:
            self.is_playing = True
            self.playback_thread = threading.Thread(
                target=self._playback_loop, daemon=True
            )
            self.playback_thread.start()

    def stop(self):
        self.is_playing, self.is_paused = False, False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=0.2)
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def toggle_pause(self):
        if self.is_playing:
            self.is_paused = not self.is_paused

    def seek(self, seconds_offset: float):
        if self.song and self.audio_data is not None:
            frame_offset = int(seconds_offset * self.song.frame_rate)
            new_frame = self.current_frame + frame_offset
            self.seek_to_frame(new_frame)

    def seek_to_frame(self, target_frame: int):
        if self.song and self.audio_data is not None:
            with self.playback_lock:
                safe_target_frame = max(
                    0, min(target_frame, len(self.audio_data) - CHUNK_SIZE)
                )
                self.current_frame = safe_target_frame
                self._initialize_filter_states()

    def update_eq(self, gains_db: List[float], q_factor: float):
        if not self.song:
            return
        with self.playback_lock:
            for i, gain in enumerate(gains_db):
                if abs(gain) < 0.1:
                    self.filters[i], self.filter_states[i] = None, None
                else:
                    self.filters[i] = design_peaking_filter(
                        FREQ_BANDS[i], q_factor, gain, self.song.frame_rate
                    )
                    if self.filter_states[i] is None and self.song:
                        self.filter_states[i] = np.zeros((self.song.channels, 2))

    def _initialize_filter_states(self):
        if not self.song:
            return
        self.filter_states = []
        for filter_coeffs in self.filters:
            if filter_coeffs is not None:
                # Initialize filter state for each channel (2 delay elements per channel)
                self.filter_states.append(np.zeros((self.song.channels, 2)))
            else:
                self.filter_states.append(None)

    def close(self):
        self.stop()
        self.p.terminate()


class RealtimeEqualizer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Python Pro Equalizer")
        self.geometry("1000x900")  # Daha büyük pencere
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.audio_engine = AudioEngine(
            on_progress_update=self.update_ui_on_progress,
            on_visualization_update=self.update_visualization,
            on_playback_stopped=self.on_playback_stopped,
        )
        self.gains_db: List[float] = [0.0] * len(FREQ_BANDS)
        self.q_factor: float = 2.0
        self.is_seeking_with_slider = False
        self.sliders: List[ctk.CTkSlider] = []
        self.slider_value_labels: List[ctk.CTkLabel] = []
        self.q_value_label: Optional[ctk.CTkLabel] = None
        self._setup_ui()
        self.populate_library_from_catalog()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # Visualizer için
        self.grid_rowconfigure(5, weight=2)  # Kütüphane için daha fazla yer
        self._create_info_panel().grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self._create_progress_bar_panel().grid(
            row=1, column=0, padx=20, pady=0, sticky="ew"
        )
        self._create_player_controls().grid(row=2, column=0, pady=10)
        self._create_visualizer().grid(row=3, column=0, padx=20, pady=5, sticky="ewns")
        self._create_equalizer_panel().grid(
            row=4, column=0, padx=20, pady=10, sticky="ew"
        )
        self._create_library_panel().grid(
            row=5, column=0, padx=20, pady=10, sticky="nsew"
        )
        self.status_label = ctk.CTkLabel(
            self, text="Wird gestartet...", font=ctk.CTkFont(size=11)
        )
        self.status_label.grid(row=6, column=0, pady=5, sticky="ew")

    def _create_info_panel(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid_columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(
            frame,
            text="Musikplayer & Equalizer",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=15, pady=(10, 2), sticky="w")
        self.artist_label = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12))
        self.artist_label.grid(row=1, column=0, padx=15, pady=0, sticky="w")
        self.album_label = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=12))
        self.album_label.grid(row=2, column=0, padx=15, pady=(0, 10), sticky="w")
        return frame

    def _create_progress_bar_panel(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(1, weight=1)
        self.time_label = ctk.CTkLabel(
            frame, text="00:00 / 00:00", font=ctk.CTkFont(size=11)
        )
        self.time_label.grid(row=0, column=0, padx=(0, 10))
        self.progress_bar = ctk.CTkSlider(
            frame, from_=0, to=1000, command=None, state="disabled"
        )
        self.progress_bar.set(0)
        self.progress_bar.bind("<Button-1>", self.on_slider_press)
        self.progress_bar.bind("<ButtonRelease-1>", self.on_slider_release)
        self.progress_bar.grid(row=0, column=1, sticky="ew")
        return frame

    def _create_player_controls(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        self.rewind_button = ctk.CTkButton(
            frame,
            text="⏪",
            command=lambda: self.audio_engine.seek(-SEEK_SECONDS),
            state="disabled",
            font=ctk.CTkFont(size=20),
            width=60,
        )
        self.rewind_button.pack(side="left", padx=5)
        self.play_pause_button = ctk.CTkButton(
            frame,
            text="▶",
            command=self.toggle_pause,
            state="disabled",
            font=ctk.CTkFont(size=22, weight="bold"),
            width=80,
        )
        self.play_pause_button.pack(side="left", padx=5)
        self.forward_button = ctk.CTkButton(
            frame,
            text="⏩",
            command=lambda: self.audio_engine.seek(SEEK_SECONDS),
            state="disabled",
            font=ctk.CTkFont(size=20),
            width=60,
        )
        self.forward_button.pack(side="left", padx=5)
        return frame

    def _create_visualizer(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color="black", corner_radius=10)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        self.db_label_canvas = tk.Canvas(
            frame, bg="black", width=50, height=140, highlightthickness=0
        )
        self.db_label_canvas.grid(row=0, column=0, sticky="ns", pady=5, padx=(5, 0))
        self.vis_canvas = tk.Canvas(frame, bg="black", height=140, highlightthickness=0)
        self.vis_canvas.grid(row=0, column=1, sticky="nsew", pady=5, padx=(0, 5))
        self.label_canvas = tk.Canvas(
            frame, bg="black", height=25, highlightthickness=0
        )
        self.label_canvas.grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=(0, 5))
        self._draw_db_labels()
        canvas_width = 1000 - 40 - 50 - 10  # Yeni pencere genişliği için güncelle
        bar_width = canvas_width / NUM_VIS_BARS
        self.vis_bars = [
            self.vis_canvas.create_rectangle(
                i * bar_width, 140, (i + 1) * bar_width, 140, fill="#1f77b4", outline=""
            )
            for i in range(NUM_VIS_BARS)
        ]
        return frame

    def _create_equalizer_panel(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self)
        frame.grid_columnconfigure(0, weight=1)
        eq_sliders_frame = ctk.CTkFrame(frame, fg_color="transparent")
        eq_sliders_frame.grid(row=0, column=0, sticky="ew", columnspan=2, pady=(5, 10))

        self.sliders, self.slider_value_labels = [], []
        for i, label_text in enumerate(BAND_LABELS):
            slider_frame = ctk.CTkFrame(eq_sliders_frame, fg_color="transparent")
            slider_frame.pack(side="left", expand=True, fill="x")
            ctk.CTkLabel(
                slider_frame, text=label_text, font=ctk.CTkFont(size=11)
            ).pack()
            slider = ctk.CTkSlider(
                slider_frame,
                from_=-MAX_GAIN_DB,
                to=MAX_GAIN_DB,
                orientation="vertical",
                height=120,
                command=lambda v, idx=i: self.on_eq_slider_change(v, idx),
                number_of_steps=int(2 * MAX_GAIN_DB * 10),
            )
            slider.set(0)
            slider.pack(pady=(5, 0))
            self.sliders.append(slider)
            value_label = ctk.CTkLabel(
                slider_frame, text="+0.0 dB", font=ctk.CTkFont(size=11)
            )
            value_label.pack(pady=(2, 0))
            self.slider_value_labels.append(value_label)
            freq = FREQ_BANDS[i]
            freq_text = f"{freq/1000:.1f} kHz" if freq >= 1000 else f"{freq} Hz"
            ctk.CTkLabel(
                slider_frame,
                text=freq_text,
                font=ctk.CTkFont(size=10),
                text_color="gray60",
            ).pack()

        q_frame = ctk.CTkFrame(frame, fg_color="transparent")
        q_frame.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkLabel(q_frame, text="Filterschärfe (Q):").pack(side="left", padx=(0, 5))

        self.q_value_label = ctk.CTkLabel(
            q_frame, text=f"{self.q_factor:.2f}", font=ctk.CTkFont(size=11), width=35
        )
        self.q_value_label.pack(side="left")

        self.q_slider = ctk.CTkSlider(
            q_frame, from_=0.5, to=10.0, command=self.on_q_change, number_of_steps=95
        )
        self.q_slider.set(self.q_factor)
        self.q_slider.pack(side="left", padx=10)

        reset_button = ctk.CTkButton(
            frame, text="EQ Zurücksetzen", command=self.reset_eq, width=100
        )
        reset_button.grid(row=1, column=1, sticky="e", padx=10, pady=5)
        return frame

    def _create_library_panel(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(
            header, text="MP3-Bibliothek", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        ctk.CTkButton(
            header, text="Ordner scannen", command=self.scan_mp3_files_dialog, width=120
        ).pack(side="right")
        self.music_list_frame = ctk.CTkScrollableFrame(frame)
        self.music_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        return frame

    def play_mp3_file(self, filepath: str):
        if self.audio_engine.load_song(filepath):
            self.audio_engine.update_eq(self.gains_db, self.q_factor)
            self.audio_engine.play()
            self._update_song_info_panel(filepath)
            self._update_visualizer_freq_labels()
            self._enable_playback_controls()
            self.status_label.configure(
                text=f"Wird abgespielt: {os.path.basename(filepath)}"
            )

    def toggle_pause(self):
        self.audio_engine.toggle_pause()
        self.play_pause_button.configure(
            text="▶" if self.audio_engine.is_paused else "⏸"
        )

    def on_eq_slider_change(self, value: float, index: int):
        self.gains_db[index] = value
        self.audio_engine.update_eq(self.gains_db, self.q_factor)
        formatted_text = f"{value:+.1f} dB"
        if index < len(self.slider_value_labels):
            self.slider_value_labels[index].configure(text=formatted_text)

    def on_q_change(self, value: float):
        self.q_factor = value
        self.audio_engine.update_eq(self.gains_db, self.q_factor)
        if self.q_value_label:
            self.q_value_label.configure(text=f"{value:.2f}")

    def reset_eq(self):
        self.q_factor = 2.0
        self.q_slider.set(self.q_factor)
        if self.q_value_label:
            self.q_value_label.configure(text=f"{self.q_factor:.2f}")

        for i in range(len(self.sliders)):
            self.sliders[i].set(0)
            if i < len(self.slider_value_labels):
                self.slider_value_labels[i].configure(text="+0.0 dB")
            self.gains_db[i] = 0.0
        self.audio_engine.update_eq(self.gains_db, self.q_factor)

    def on_slider_press(self, _event):
        self.is_seeking_with_slider = True

    def on_slider_release(self, _event):
        if self.is_seeking_with_slider:
            self.is_seeking_with_slider = False
            if self.audio_engine.song and self.audio_engine.audio_data is not None:
                slider_value = self.progress_bar.get()
                total_frames = len(self.audio_engine.audio_data)
                target_frame = int((slider_value / 1000.0) * total_frames)
                self.audio_engine.seek_to_frame(target_frame)

    def update_ui_on_progress(self, current_frame: int, total_seconds: float):
        if self.audio_engine.song is None or self.audio_engine.audio_data is None:
            return
        current_seconds = current_frame / self.audio_engine.song.frame_rate
        self.time_label.configure(
            text=f"{_format_duration(current_seconds)} / {_format_duration(total_seconds)}"
        )
        if not self.is_seeking_with_slider and len(self.audio_engine.audio_data) > 0:
            self.progress_bar.set(
                (current_frame / len(self.audio_engine.audio_data)) * 1000
            )

    def update_visualization(self, samples: np.ndarray):
        if not self.audio_engine.is_playing or len(samples) < 1:
            return
        # FFT işlemi: pyfftw ile hızlı, mono/stereo ayrımı
        if samples.ndim > 1:
            # Stereo ise sadece sol kanalı göster
            samples = samples[:, 0]
        windowed = samples * np.hanning(len(samples))
        fft_result = pyfftw.interfaces.numpy_fft.rfft(windowed)
        magnitudes = 20 * np.log10(np.abs(fft_result) + 1e-9)
        canvas_height = self.vis_canvas.winfo_height()
        if canvas_height <= 1:
            canvas_height = 140  # Yeni yükseklik
        num_fft_bins, log_indices = len(magnitudes), np.logspace(
            0, np.log10(len(magnitudes) - 1), NUM_VIS_BARS + 1, dtype=int
        )
        for i in range(NUM_VIS_BARS):
            start_idx, end_idx = log_indices[i], log_indices[i + 1]
            bar_magnitude = (
                np.mean(magnitudes[start_idx:end_idx])
                if start_idx < end_idx
                else (
                    magnitudes[start_idx]
                    if start_idx < num_fft_bins
                    else VIS_MIN_DB - 10
                )
            )
            normalized_height = (bar_magnitude - VIS_MIN_DB) / VIS_DB_RANGE
            bar_height = min(max(normalized_height * canvas_height, 0), canvas_height)
            if len(self.vis_canvas.coords(self.vis_bars[i])) >= 4:
                x0, _, x1, _ = self.vis_canvas.coords(self.vis_bars[i])
                self.vis_canvas.coords(
                    self.vis_bars[i], x0, canvas_height - bar_height, x1, canvas_height
                )
            color = "#1f77b4"
            if bar_magnitude > -12.0:
                color = "#2ca02c"
            if bar_magnitude > -6.0:
                color = "#ff7f0e"
            if bar_magnitude > -3.0:
                color = "#d62728"
            if bar_magnitude > 0.0:
                color = "#8b0000"
            self.vis_canvas.itemconfigure(self.vis_bars[i], fill=color)

    def on_playback_stopped(self):
        self.after(0, self._disable_playback_controls_if_needed)

    def _enable_playback_controls(self):
        self.play_pause_button.configure(state="normal", text="⏸")
        self.rewind_button.configure(state="normal")
        self.forward_button.configure(state="normal")
        self.progress_bar.configure(state="normal")

    def _disable_playback_controls_if_needed(self):
        if not self.audio_engine.is_playing:
            self.play_pause_button.configure(state="disabled", text="▶")
            self.rewind_button.configure(state="disabled")
            self.forward_button.configure(state="disabled")
            self.progress_bar.configure(state="disabled")
            self.progress_bar.set(0)
            self.time_label.configure(text="00:00 / 00:00")
            self.status_label.configure(
                text="Wiedergabe beendet. Wählen Sie einen Song."
            )

    def populate_library_from_catalog(self, directory: Optional[str] = None):
        if directory is None:
            music_directory = resource_path("music")
            directory_to_scan = music_directory
        else:
            directory_to_scan = directory

        for widget in self.music_list_frame.winfo_children():
            widget.destroy()

        found_files = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(directory_to_scan, "*.mp3"))
        }

        matched_count = 0
        for entry in MUSIC_CATALOG:
            if entry["filename"] in found_files:
                entry["filepath"] = found_files[entry["filename"]]
                matched_count += 1
            else:
                entry["filepath"] = None
            self._create_library_entry_widget(entry)
            display_name = (
                os.path.basename(directory) if directory else "Interne Bibliothek"
            )
            self.music_list_frame.configure(
                label_text=f"Musikbibliothek ({display_name})"
            )
            self.status_label.configure(
                text=f"{matched_count} / {len(MUSIC_CATALOG)} Werke gefunden."
            )

    def _create_library_entry_widget(self, entry: Dict[str, Any]):
        entry_frame = ctk.CTkFrame(self.music_list_frame, fg_color=("gray85", "gray20"))
        entry_frame.pack(fill="x", expand=True, pady=3, padx=5)
        entry_frame.grid_columnconfigure(0, weight=1)

        # Ana bilgi frame'i
        info_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        info_frame.grid(row=0, column=0, padx=10, pady=8, sticky="ew")

        # Başlık daha büyük ve bold
        title_text = f"{entry['composer']}: {entry['title']}"
        ctk.CTkLabel(
            info_frame,
            text=title_text,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(fill="x")

        # Açıklama daha küçük ama okunabilir
        ctk.CTkLabel(
            info_frame,
            text=entry["description"],
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify=tk.LEFT,
            wraplength=700,  # Daha geniş wrap
            text_color=("gray10", "gray70"),  # Daha soft renk
        ).pack(fill="x", pady=(3, 0))

        # Buton frame'i
        button_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, padx=10, pady=8)
        button_state = "normal" if entry["filepath"] else "disabled"
        button_text = "▶ Spielen" if entry["filepath"] else "❌ Datei fehlt"
        play_button = ctk.CTkButton(
            button_frame,
            text=button_text,
            width=110,
            height=32,
            state=button_state,
            font=ctk.CTkFont(size=12),
            command=lambda fp=entry["filepath"]: self.play_mp3_file(fp) if fp else None,
        )
        play_button.pack(expand=True)

    def scan_mp3_files_dialog(self):
        directory = filedialog.askdirectory()
        if directory:
            self.populate_library_from_catalog(directory)

    def _draw_db_labels(self):
        self.db_label_canvas.delete("all")
        canvas_height = 140  # Yeni yükseklik

        # Daha uygun aralıklarla dB değerleri göster
        db_values = [12, 0, -12, -24, -36, -48]

        for db_val in db_values:
            y_pos = canvas_height - (
                ((db_val - VIS_MIN_DB) / VIS_DB_RANGE) * canvas_height
            )

            if 0 <= y_pos <= canvas_height:
                # 0 dB özel renk ve kalınlık
                if db_val == 0:
                    color = "white"
                    font_style = ("Arial", 9, "bold")
                    # 0 dB için çizgi ekle
                    self.db_label_canvas.create_line(
                        35, y_pos, 50, y_pos, fill="white", width=2
                    )
                else:
                    color = "lightgray"
                    font_style = ("Arial", 8)

                # Pozitif değerler için + işareti ekle
                text = f"+{db_val}" if db_val > 0 else str(db_val)

                self.db_label_canvas.create_text(
                    25,
                    y_pos,
                    text=text,
                    fill=color,
                    font=font_style,
                    anchor="center",
                )

    def _update_visualizer_freq_labels(self):
        self.label_canvas.delete("all")
        if not self.audio_engine.song:
            return

        canvas_width = 1000 - 40 - 50 - 10  # Yeni pencere genişliği için güncelle
        bar_width = canvas_width / NUM_VIS_BARS
        fft_freqs = np.fft.rfftfreq(CHUNK_SIZE, 1 / self.audio_engine.song.frame_rate)
        log_indices = np.logspace(
            0, np.log10(len(fft_freqs) - 1), NUM_VIS_BARS + 1, dtype=int
        )

        for i in range(0, NUM_VIS_BARS, 4):
            idx = log_indices[i]
            if idx < len(fft_freqs):
                freq = fft_freqs[idx]
                text = f"{freq/1000:.1f}k" if freq >= 1000 else f"{int(freq)}"
                x_pos = (i + 0.5) * bar_width
                self.label_canvas.create_text(
                    x_pos, 12, text=text, fill="lightgray", font=("Arial", 9)
                )

    def _update_song_info_panel(self, filepath: str):
        try:
            audio = MP3(filepath, ID3=ID3)
            title = str(audio.get("TIT2", os.path.basename(filepath)))
            artist = str(audio.get("TPE1", "Unbekannt"))
            album = str(audio.get("TALB", "Unbekannt"))
            year = str(audio.get("TDRC", "-").text[0] if audio.get("TDRC") else "-")
        except Exception:
            title, artist, album, year = (
                os.path.basename(filepath),
                "Unbekannt",
                "Unbekannt",
                "-",
            )
        self.title_label.configure(text=title)
        self.artist_label.configure(text=f"Künstler: {artist}")
        self.album_label.configure(text=f"Album: {album} ({year})")

    def on_closing(self):
        self.audio_engine.close()
        self.destroy()


if __name__ == "__main__":
    app = RealtimeEqualizer()
    app.mainloop()
