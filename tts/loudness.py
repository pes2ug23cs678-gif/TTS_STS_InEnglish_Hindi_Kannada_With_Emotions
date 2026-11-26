"""Loudness normalisation for synthesised speech.

Runs the MP3 bytes through ffmpeg's loudnorm filter so that quiet clips are raised to a
consistent, louder level. If ffmpeg is missing or fails, the original audio is returned
unchanged, so speech is never lost because of this step.
"""
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

TARGET_LUFS = -12.0      # integrated loudness target; -14 is the common streaming level, -12 is louder
TRUE_PEAK_DB = -1.5      # ceiling for true peaks, so the louder output does not clip
LOUDNESS_RANGE = 11.0    # keeps the dynamic range natural rather than squashed
BITRATE = "128k"

# OFF by default: Edge/Parler audio is already at a fine level, and piping MP3 through
# ffmpeg's loudnorm can produce a valid-but-silent/garbled file on some ffmpeg builds
# (a source of "the player works but there's no sound"). Enable only if you want the
# louder, consistent target: STS_ENABLE_LOUDNORM=1.
_ENABLE_LOUDNORM = os.getenv("STS_ENABLE_LOUDNORM", "0") == "1"


def normalize_loudness(mp3_bytes: bytes, target_lufs: float = TARGET_LUFS) -> bytes:
    """Returns MP3 bytes at the target loudness, or the original bytes if the step is
    disabled or cannot run. Disabled by default so the raw (audible) TTS audio is used."""
    if not mp3_bytes:
        return mp3_bytes
    if not _ENABLE_LOUDNORM:
        return mp3_bytes
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; loudness normalisation skipped.")
        return mp3_bytes
    filter_chain = f"loudnorm=I={target_lufs}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
           "-af", filter_chain, "-f", "mp3", "-b:a", BITRATE, "pipe:1"]
    try:
        result = subprocess.run(cmd, input=mp3_bytes, capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("Loudness normalisation failed (%s); using original audio.", err)
        return mp3_bytes
    if result.returncode != 0 or not result.stdout:
        logger.warning("Loudness normalisation returned an error (%s); using original audio.",
                       result.stderr.decode("utf-8", "replace")[:200])
        return mp3_bytes
    return result.stdout
