import os
import subprocess
import shutil
import tempfile
import folder_paths
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Helper: locate ffmpeg
# ---------------------------------------------------------------------------

def get_ffmpeg_path():
    """Return the path to the ffmpeg binary or raise an error."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise EnvironmentError(
            "ffmpeg not found on PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )
    return path


def get_ffprobe_path():
    """Return the path to the ffprobe binary or raise an error."""
    path = shutil.which("ffprobe")
    if path is None:
        raise EnvironmentError(
            "ffprobe not found on PATH. "
            "Please install ffmpeg (includes ffprobe): https://ffmpeg.org/download.html"
        )
    return path


# ---------------------------------------------------------------------------
# Ensure an output directory exists
# ---------------------------------------------------------------------------

OUTPUT_DIR = folder_paths.get_output_directory()

def _ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Node 1 – Combine Video + Audio (file‑path based)
# ---------------------------------------------------------------------------

class CombineVideoAudio:
    """
    Merges a video file and an audio file into a single output file using
    ffmpeg.  Accepts file paths (strings) that come from other ComfyUI nodes
    such as video‑save or audio‑save nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to the input video file."
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to the input audio file."
                }),
                "output_filename": ("STRING", {
                    "default": "combined_output",
                    "multiline": False,
                    "tooltip": "Name of the output file (without extension)."
                }),
                "output_format": (["mp4", "mkv", "mov", "webm", "avi"], {
                    "default": "mp4",
                    "tooltip": "Container format for the output file."
                }),
                "video_codec": (["copy", "libx264", "libx265", "libvpx-vp9", "mpeg4"], {
                    "default": "copy",
                    "tooltip": "'copy' keeps original encoding (fastest). Others re‑encode."
                }),
                "audio_codec": (["copy", "aac", "libmp3lame", "libvorbis", "libopus", "pcm_s16le"], {
                    "default": "aac",
                    "tooltip": "'copy' keeps original encoding. Others re‑encode."
                }),
            },
            "optional": {
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Multiply the audio volume (1.0 = unchanged)."
                }),
                "shortest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stop when the shortest stream ends."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite output file if it already exists."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "combine"
    CATEGORY = "video/audio"
    OUTPUT_NODE = True
    DESCRIPTION = "Combine a video file and an audio file into one file using ffmpeg."

    def combine(
        self,
        video_path: str,
        audio_path: str,
        output_filename: str,
        output_format: str,
        video_codec: str,
        audio_codec: str,
        audio_volume: float = 1.0,
        shortest: bool = True,
        overwrite: bool = True,
    ):
        # ---- validate inputs ------------------------------------------------
        video_path = video_path.strip().strip('"').strip("'")
        audio_path = audio_path.strip().strip('"').strip("'")

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ffmpeg = get_ffmpeg_path()

        _ensure_dir(OUTPUT_DIR)
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{output_filename}.{output_format}"
        )

        # ---- build ffmpeg command -------------------------------------------
        cmd = [
            ffmpeg,
            "-i", video_path,
            "-i", audio_path,
        ]

        # Map streams explicitly
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

        # Video codec
        cmd += ["-c:v", video_codec]

        # Audio codec & optional volume filter
        if audio_volume != 1.0:
            # When we apply a filter we must re‑encode – ignore "copy"
            effective_audio_codec = audio_codec if audio_codec != "copy" else "aac"
            cmd += [
                "-af", f"volume={audio_volume}",
                "-c:a", effective_audio_codec,
            ]
        else:
            cmd += ["-c:a", audio_codec]

        if shortest:
            cmd += ["-shortest"]

        if overwrite:
            cmd += ["-y"]

        cmd += [output_path]

        # ---- run ffmpeg -----------------------------------------------------
        print(f"[CombineVideoAudio] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
            )

        print(f"[CombineVideoAudio] Output saved to: {output_path}")
        return (output_path,)


# ---------------------------------------------------------------------------
# Node 2 – Combine IMAGE sequence + Audio → video file
# ---------------------------------------------------------------------------

class CombineImageSequenceAudio:
    """
    Takes a batch of IMAGE frames (the standard ComfyUI image tensor) and an
    audio file, then produces a combined video with audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to the input audio file."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.5,
                    "tooltip": "Frames per second for the output video."
                }),
                "output_filename": ("STRING", {
                    "default": "frames_audio_combined",
                    "multiline": False,
                }),
                "output_format": (["mp4", "mkv", "mov", "webm"], {
                    "default": "mp4",
                }),
                "video_codec": (["libx264", "libx265", "libvpx-vp9", "mpeg4"], {
                    "default": "libx264",
                }),
                "audio_codec": (["aac", "libmp3lame", "libvorbis", "libopus", "copy"], {
                    "default": "aac",
                }),
                "quality_crf": ("INT", {
                    "default": 19,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "CRF value (lower = better quality / bigger file)."
                }),
            },
            "optional": {
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                }),
                "shortest": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "combine"
    CATEGORY = "video/audio"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Convert an IMAGE batch (frames) + an audio file into a single "
        "video with audio using ffmpeg."
    )

    def combine(
        self,
        images: torch.Tensor,
        audio_path: str,
        fps: float,
        output_filename: str,
        output_format: str,
        video_codec: str,
        audio_codec: str,
        quality_crf: int,
        audio_volume: float = 1.0,
        shortest: bool = True,
    ):
        audio_path = audio_path.strip().strip('"').strip("'")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ffmpeg = get_ffmpeg_path()
        _ensure_dir(OUTPUT_DIR)

        # images: (B, H, W, C)  float32 0‑1
        if images.ndim != 4:
            raise ValueError(f"Expected 4‑D image tensor (B,H,W,C), got {images.shape}")

        B, H, W, C = images.shape

        # Write raw frames via pipe (rgb24)
        output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.{output_format}")

        cmd = [
            ffmpeg,
            # input: raw video from pipe
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(fps),
            "-i", "pipe:0",
            # input: audio file
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", video_codec,
            "-crf", str(quality_crf),
            "-pix_fmt", "yuv420p",
        ]

        if audio_volume != 1.0:
            effective_audio_codec = audio_codec if audio_codec != "copy" else "aac"
            cmd += ["-af", f"volume={audio_volume}", "-c:a", effective_audio_codec]
        else:
            cmd += ["-c:a", audio_codec]

        if shortest:
            cmd += ["-shortest"]

        cmd += ["-y", output_path]

        print(f"[CombineImageSequenceAudio] Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Feed frames into ffmpeg stdin
        for i in range(B):
            frame_np = images[i].cpu().numpy()  # (H, W, C) float32 0‑1
            frame_uint8 = (np.clip(frame_np, 0, 1) * 255).astype(np.uint8)
            process.stdin.write(frame_uint8.tobytes())

        process.stdin.close()
        _, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {process.returncode}):\n{stderr.decode()}"
            )

        print(f"[CombineImageSequenceAudio] Output saved to: {output_path}")
        return (output_path,)


# ---------------------------------------------------------------------------
# Node 3 – Replace / Remove Audio from an existing video
# ---------------------------------------------------------------------------

class ReplaceAudioInVideo:
    """
    Strip the existing audio from a video and optionally mux in a new audio
    track.  If *audio_path* is left empty the output will be silent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "output_filename": ("STRING", {
                    "default": "replaced_audio",
                    "multiline": False,
                }),
                "output_format": (["mp4", "mkv", "mov", "webm", "avi"], {
                    "default": "mp4",
                }),
            },
            "optional": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty to produce a silent video."
                }),
                "video_codec": (["copy", "libx264", "libx265"], {
                    "default": "copy",
                }),
                "audio_codec": (["aac", "libmp3lame", "copy"], {
                    "default": "aac",
                }),
                "shortest": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "replace_audio"
    CATEGORY = "video/audio"
    OUTPUT_NODE = True
    DESCRIPTION = "Replace or remove the audio track in a video file."

    def replace_audio(
        self,
        video_path: str,
        output_filename: str,
        output_format: str,
        audio_path: str = "",
        video_codec: str = "copy",
        audio_codec: str = "aac",
        shortest: bool = True,
    ):
        video_path = video_path.strip().strip('"').strip("'")
        audio_path = audio_path.strip().strip('"').strip("'")

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ffmpeg = get_ffmpeg_path()
        _ensure_dir(OUTPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.{output_format}")

        if audio_path and os.path.isfile(audio_path):
            # Replace audio
            cmd = [
                ffmpeg,
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", video_codec,
                "-c:a", audio_codec,
            ]
            if shortest:
                cmd += ["-shortest"]
            cmd += ["-y", output_path]
        else:
            # Strip audio entirely
            cmd = [
                ffmpeg,
                "-i", video_path,
                "-map", "0:v:0",
                "-c:v", video_codec,
                "-an",
                "-y", output_path,
            ]

        print(f"[ReplaceAudioInVideo] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")

        print(f"[ReplaceAudioInVideo] Output saved to: {output_path}")
        return (output_path,)


# ---------------------------------------------------------------------------
# Node 4 – Get media info (duration, codecs, resolution …)
# ---------------------------------------------------------------------------

class GetMediaInfo:
    """Return basic information about a media file using ffprobe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT",)
    RETURN_NAMES = ("info_text", "duration_seconds", "width", "height",)
    FUNCTION = "probe"
    CATEGORY = "video/audio"
    DESCRIPTION = "Retrieve duration, resolution, and codec info via ffprobe."

    def probe(self, file_path: str):
        file_path = file_path.strip().strip('"').strip("'")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ffprobe = get_ffprobe_path()

        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

        import json
        data = json.loads(result.stdout)

        duration = float(data.get("format", {}).get("duration", 0))
        width = 0
        height = 0

        info_lines = [f"File: {file_path}"]
        info_lines.append(f"Duration: {duration:.2f}s")

        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "unknown")
            codec_name = stream.get("codec_name", "unknown")
            if codec_type == "video":
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                fps = stream.get("r_frame_rate", "?")
                info_lines.append(
                    f"Video: {codec_name} {width}x{height} @ {fps} fps"
                )
            elif codec_type == "audio":
                sr = stream.get("sample_rate", "?")
                channels = stream.get("channels", "?")
                info_lines.append(
                    f"Audio: {codec_name} {sr}Hz {channels}ch"
                )

        info_text = "\n".join(info_lines)
        print(f"[GetMediaInfo]\n{info_text}")

        return (info_text, duration, width, height)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CombineVideoAudio":          CombineVideoAudio,
    "CombineImageSequenceAudio":  CombineImageSequenceAudio,
    "ReplaceAudioInVideo":        ReplaceAudioInVideo,
    "GetMediaInfo":               GetMediaInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineVideoAudio":          "🎬 Combine Video + Audio",
    "CombineImageSequenceAudio":  "🎞️ Frames + Audio → Video",
    "ReplaceAudioInVideo":        "🔇 Replace / Remove Audio",
    "GetMediaInfo":               "ℹ️ Get Media Info",
}