"""
Alarm utilities.

Generates a short sine-wave beep as a .wav file using only the Python
standard library (no audio assets to ship), and plays it on a background
thread so the video processing loop is never blocked while it sounds.

Playback intentionally uses each OS's *built-in* audio player instead of
a third-party pip package: `winsound` (stdlib, Windows), `afplay`
(ships with macOS), or `aplay`/`paplay` (ships with most Linux desktops
via ALSA/PulseAudio). This means there is nothing to compile and nothing
extra to pip install for sound to work on a normal desktop/laptop.

If none of these are available (e.g. a headless server with no audio
device — which has no speakers to play sound through anyway), playback
degrades gracefully to a no-op. The on-screen visual alert always still
works regardless.
"""

import math
import os
import platform
import shutil
import struct
import subprocess
import threading
import wave


def _linux_player_cmd(path: str):
    for player in ("paplay", "aplay"):
        exe = shutil.which(player)
        if exe:
            return [exe, path] if player == "paplay" else [exe, "-q", path]
    return None


def is_audio_available() -> bool:
    """Best-effort check for whether *some* playback backend exists."""
    system = platform.system()
    if system == "Windows":
        try:
            import winsound  # noqa: F401
            return True
        except ImportError:
            return False
    if system == "Darwin":
        return shutil.which("afplay") is not None
    return _linux_player_cmd("dummy") is not None


def generate_alarm_wav(
    path: str,
    frequency: int = 1000,
    duration: float = 1.0,
    volume: float = 0.5,
    sample_rate: int = 44100,
) -> str:
    """Create a simple sine-wave beep .wav file if it doesn't already exist."""
    if os.path.exists(path):
        return path

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_samples = int(sample_rate * duration)
    fade_samples = max(1, int(sample_rate * 0.05))  # 50ms fade to avoid a click

    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            fade = min(1.0, (n_samples - i) / fade_samples, (i + 1) / fade_samples)
            sample = volume * fade * math.sin(2 * math.pi * frequency * t)
            frames += struct.pack("<h", int(sample * 32767))
        wav_file.writeframes(frames)

    return path


_playback_lock = threading.Lock()
_currently_playing = False


def _play_blocking(path: str) -> None:
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    elif system == "Darwin":
        subprocess.run(["afplay", path], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        cmd = _linux_player_cmd(path)
        if cmd is None:
            raise RuntimeError("no Linux audio player (aplay/paplay) found on PATH")
        subprocess.run(cmd, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_alarm_async(path: str) -> None:
    """
    Play the alarm sound on a background thread without blocking the
    caller. If the alarm is already playing, this call is a no-op so the
    alarm doesn't overlap/stack on every frame.
    """
    global _currently_playing

    with _playback_lock:
        if _currently_playing:
            return
        _currently_playing = True

    def _worker():
        global _currently_playing
        try:
            _play_blocking(path)
        except Exception as exc:  # pragma: no cover - depends on host audio device
            print(f"[alarm] could not play sound ({exc}). Visual alert still active.")
        finally:
            with _playback_lock:
                _currently_playing = False

    threading.Thread(target=_worker, daemon=True).start()
