# ComfyUI – Combine Video & Audio

A set of custom nodes for **ComfyUI** that merge video and audio using
**ffmpeg**.

## Prerequisites

| Dependency | How to install |
|------------|----------------|
| **ffmpeg** (+ ffprobe) | `sudo apt install ffmpeg` · `brew install ffmpeg` · or download from <https://ffmpeg.org/download.html> and add to PATH |

## Nodes

### 🎬 Combine Video + Audio
Merge two files (a video and an audio track) into one output file.

| Input | Type | Note |
|-------|------|------|
| `video_path` | STRING | Path to video file |
| `audio_path` | STRING | Path to audio file |
| `output_filename` | STRING | Name without extension |
| `output_format` | mp4 / mkv / mov / webm / avi | |
| `video_codec` | copy / libx264 / … | `copy` = no re‑encode |
| `audio_codec` | copy / aac / … | |
| `audio_volume` | FLOAT | 1.0 = original |
| `shortest` | BOOL | Trim to shortest stream |

**Output:** `output_video_path` (STRING)

---

### 🎞️ Frames + Audio → Video
Convert a ComfyUI **IMAGE** batch (frames) + audio file into a video.

| Input | Type |
|-------|------|
| `images` | IMAGE (B,H,W,C) |
| `audio_path` | STRING |
| `fps` | FLOAT |
| `quality_crf` | INT (0‑51) |

**Output:** `output_video_path` (STRING)

---

### 🔇 Replace / Remove Audio
Strip the existing audio from a video and optionally add a new track.
Leave `audio_path` empty to produce a silent video.

---

### ℹ️ Get Media Info
Uses **ffprobe** to return duration, resolution, and codec details.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CrazyDashTool/Combine-video-and-audio-comfyUI
# Restart ComfyUI
