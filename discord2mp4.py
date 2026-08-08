import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from fractions import Fraction


MAX_SIZE_MIB = 8.0
TARGET_SIZE_MIB = 7.75  # Leave room for MP4/container overhead.
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
DEFAULT_BPP = 0.045  # Bits per pixel per frame target for x264 slow.
CROP_SAMPLES = 5
CROP_SAMPLE_SECONDS = 2.0


def run(cmd, capture=True, check=True):
    kwargs = {"text": True, "check": check}
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, **kwargs)


def require_tools():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required tool(s): " + ", ".join(missing) +
            ". Install FFmpeg and make sure ffmpeg/ffprobe are in PATH."
        )


def parse_fraction(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json",
        path,
    ]
    data = json.loads(run(cmd).stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("No video stream found.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    fps = parse_fraction(video.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = parse_fraction(video.get("r_frame_rate") or "0/1")

    duration = float(data.get("format", {}).get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions.")
    if duration <= 0:
        raise ValueError("Could not determine video duration.")
    if fps <= 0:
        fps = 30.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "has_audio": has_audio,
    }


def even(value):
    value = max(2, int(round(value)))
    return value if value % 2 == 0 else value - 1


def clamp_crop(crop, src_w, src_h):
    w, h, x, y = crop
    x = max(0, min(int(x) - int(x) % 2, src_w - 2))
    y = max(0, min(int(y) - int(y) % 2, src_h - 2))
    w = even(min(w, src_w - x))
    h = even(min(h, src_h - y))
    if w <= 0 or h <= 0:
        return None
    return w, h, x, y


def detect_crop_at(path, start, sample_seconds):
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
        "-ss", f"{start:.3f}", "-i", path,
        "-t", f"{sample_seconds:.3f}",
        "-an", "-sn",
        "-vf", "cropdetect=24:2:0",
        "-f", "null", os.devnull,
    ]
    result = run(cmd, capture=True, check=False)
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr or "")
    if not matches:
        return None
    # cropdetect converges as frames are processed; the final value is usually the useful one.
    w, h, x, y = map(int, matches[-1])
    return w, h, x, y


def detect_stable_crop(path, duration, src_w, src_h):
    if duration <= 0:
        return None

    sample_len = min(CROP_SAMPLE_SECONDS, max(0.5, duration / 8.0))
    if duration <= sample_len + 0.1:
        starts = [0.0]
    else:
        # Avoid relying only on the first/last frame, where fades are common.
        fractions = [0.10, 0.30, 0.50, 0.70, 0.90]
        starts = [max(0.0, min(duration - sample_len, duration * f - sample_len / 2)) for f in fractions]

    crops = []
    for start in starts:
        crop = detect_crop_at(path, start, sample_len)
        if crop:
            crop = clamp_crop(crop, src_w, src_h)
            if crop:
                crops.append(crop)

    if not crops:
        return None

    best_crop, count = Counter(crops).most_common(1)[0]
    required = 1 if len(crops) == 1 else max(3, math.ceil(len(crops) * 0.70))
    if count < required:
        return None

    w, h, x, y = best_crop

    # Do not crop tiny edge differences caused by compression noise.
    removed_area = 1.0 - (w * h) / (src_w * src_h)
    edge_removed = (x + y + (src_w - (x + w)) + (src_h - (y + h)))
    if removed_area < 0.005 or edge_removed < 8:
        return None

    return best_crop


def fit_inside(width, height, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    scale = min(1.0, max_width / width, max_height / height)
    return even(width * scale), even(height * scale)


def scaled_dimensions(width, height, scale):
    return even(width * scale), even(height * scale)


def choose_audio_bitrate(total_kbps, has_audio, force_no_audio=False):
    if force_no_audio or not has_audio:
        return 0
    if total_kbps >= 1200:
        return 128
    if total_kbps >= 700:
        return 96
    if total_kbps >= 400:
        return 64
    if total_kbps >= 250:
        return 48
    if total_kbps >= 150:
        return 40
    if total_kbps >= 100:
        return 32
    if total_kbps >= 70:
        return 24
    if total_kbps >= 45:
        return 16
    return 0


def build_resolution_candidates(width, height):
    base_w, base_h = fit_inside(width, height)
    scales = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.42, 0.35, 0.28, 0.22, 0.17]
    candidates = []
    seen = set()
    for scale in scales:
        w, h = scaled_dimensions(base_w, base_h, scale)
        # Keep absurdly small dimensions as a last resort, but not below 160 px on the long side.
        if max(w, h) < 160:
            continue
        key = (w, h)
        if key not in seen:
            seen.add(key)
            candidates.append(key)
    if not candidates:
        candidates.append((base_w, base_h))
    return candidates


def build_fps_candidates(source_fps):
    source_fps = min(max(source_fps, 1.0), 60.0)
    raw = [source_fps, 30.0, 24.0, 20.0, 15.0, 12.0]
    result = []
    for fps in raw:
        fps = min(fps, source_fps)
        if fps < 1:
            continue
        if not any(abs(fps - old) < 0.01 for old in result):
            result.append(fps)
    return result


def choose_geometry(width, height, source_fps, video_kbps, target_bpp=DEFAULT_BPP):
    resolutions = build_resolution_candidates(width, height)
    source_fps = min(max(source_fps, 1.0), 60.0)

    # 30 FPS is the preferred floor for ordinary video. Keep >30 FPS only when
    # the bitrate is generous enough at the largest resolution.
    if source_fps > 30.0:
        base_w, base_h = resolutions[0]
        source_bpp = (video_kbps * 1000.0) / (base_w * base_h * source_fps)
        preferred_fps = source_fps if source_bpp >= target_bpp * 1.25 else 30.0
    else:
        preferred_fps = source_fps

    fps_values = [preferred_fps]
    for candidate in (30.0, 24.0, 20.0, 15.0, 12.0):
        candidate = min(candidate, source_fps)
        if candidate >= 1.0 and not any(abs(candidate - old) < 0.01 for old in fps_values):
            fps_values.append(candidate)

    # Prefer keeping temporal smoothness, then take the largest resolution that
    # has enough bits per pixel per frame. Only reduce FPS when even small
    # resolutions would otherwise be starved of bitrate.
    for fps in fps_values:
        for w, h in resolutions:
            bpp = (video_kbps * 1000.0) / (w * h * fps)
            if bpp >= target_bpp:
                return w, h, fps, bpp

    # Extreme case: use the smallest resolution and lowest available FPS.
    w, h = resolutions[-1]
    fps = fps_values[-1]
    bpp = (video_kbps * 1000.0) / (w * h * fps)
    return w, h, fps, bpp


def make_filter(crop, width, height):
    filters = []
    if crop:
        w, h, x, y = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.append(f"scale={width}:{height}:flags=lanczos")
    return ",".join(filters)


def format_fps(fps):
    if abs(fps - round(fps)) < 0.001:
        return str(int(round(fps)))
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def encode_two_pass(input_file, output_file, vf, fps, video_kbps, audio_kbps, preset="slow"):
    with tempfile.TemporaryDirectory(prefix="discord2mp4_") as tmpdir:
        passlog = os.path.join(tmpdir, "ffmpeg2pass")

        common = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_file,
            "-map", "0:v:0",
            "-vf", vf,
            "-r", format_fps(fps),
            "-c:v", "libx264",
            "-preset", preset,
            "-b:v", f"{max(2, int(video_kbps))}k",
            "-pix_fmt", "yuv420p",
            "-passlogfile", passlog,
        ]

        first_pass = common + [
            "-pass", "1",
            "-an", "-sn", "-dn",
            "-f", "null", os.devnull,
        ]
        run(first_pass, capture=True, check=True)

        second_pass = common + ["-pass", "2"]
        if audio_kbps > 0:
            second_pass += [
                "-map", "0:a:0?",
                "-c:a", "aac",
                "-b:a", f"{audio_kbps}k",
            ]
            if audio_kbps <= 32:
                second_pass += ["-ac", "1"]
        else:
            second_pass += ["-an"]

        second_pass += [
            "-sn", "-dn",
            "-map_metadata", "-1",
            "-movflags", "+faststart",
            output_file,
        ]
        run(second_pass, capture=True, check=True)


def human_mib(size_bytes):
    return size_bytes / (1024 * 1024)


def process_video(input_file, no_crop=False, no_audio=False, preset="slow", target_bpp=DEFAULT_BPP):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    require_tools()
    info = probe_video(input_file)
    src_w = info["width"]
    src_h = info["height"]
    src_fps = info["fps"]
    duration = info["duration"]

    print(f"Input: {src_w}x{src_h} @ {src_fps:.3f} FPS, {duration:.2f} s")

    crop = None
    if not no_crop:
        print("Detecting stable black borders...")
        crop = detect_stable_crop(input_file, duration, src_w, src_h)

    if crop:
        crop_w, crop_h, crop_x, crop_y = crop
        work_w, work_h = crop_w, crop_h
        print(f"Crop: {crop_w}x{crop_h}+{crop_x}+{crop_y}")
    else:
        work_w, work_h = src_w, src_h
        print("Crop: none")

    max_size_bytes = int(MAX_SIZE_MIB * 1024 * 1024)
    target_size_bytes = int(TARGET_SIZE_MIB * 1024 * 1024)

    # Reserve a little bandwidth for MP4/container overhead. The final size is also verified.
    total_target_kbps = (target_size_bytes * 8.0 / duration) / 1000.0
    usable_kbps = total_target_kbps * 0.985
    audio_kbps = choose_audio_bitrate(usable_kbps, info["has_audio"], no_audio)
    video_kbps = max(2, int(usable_kbps - audio_kbps))

    width, height, out_fps, bpp = choose_geometry(
        work_w, work_h, src_fps, video_kbps, target_bpp=target_bpp
    )

    vf = make_filter(crop, width, height)
    base_name, _ = os.path.splitext(input_file)
    output_file = f"{base_name}_discord.mp4"

    print(f"Target size: <= {MAX_SIZE_MIB:.2f} MiB (aiming for {TARGET_SIZE_MIB:.2f} MiB)")
    print(f"Video: {width}x{height} @ {out_fps:.3f} FPS, {video_kbps} kbps")
    if audio_kbps:
        print(f"Audio: AAC {audio_kbps} kbps")
    else:
        print("Audio: disabled")
    print(f"Estimated bitrate density: {bpp:.4f} bits/pixel/frame")

    attempts = 0
    current_video_kbps = video_kbps
    current_audio_kbps = audio_kbps

    while attempts < 4:
        attempts += 1
        print(f"Encoding pass set {attempts}...")
        encode_two_pass(
            input_file,
            output_file,
            vf,
            out_fps,
            current_video_kbps,
            current_audio_kbps,
            preset=preset,
        )

        actual_size = os.path.getsize(output_file)
        print(f"Result size: {human_mib(actual_size):.3f} MiB")

        if actual_size <= max_size_bytes:
            print(f"Done: {output_file}")
            return output_file

        # Recalculate the video bitrate from the measured overshoot.
        ratio = target_size_bytes / actual_size
        new_video = int((current_video_kbps + current_audio_kbps) * ratio * 0.97 - current_audio_kbps)

        if new_video >= current_video_kbps:
            new_video = current_video_kbps - max(1, int(current_video_kbps * 0.05))

        if new_video < 2 and current_audio_kbps > 0:
            print("Still too large: dropping audio and retrying.")
            current_audio_kbps = 0
            new_video = max(2, int((target_size_bytes * 8.0 / duration) / 1000.0 * 0.96))

        current_video_kbps = max(2, new_video)
        print(f"Too large; retrying at {current_video_kbps} kbps video bitrate...")

    actual_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    if actual_size > max_size_bytes:
        raise RuntimeError(
            f"Could not get below {MAX_SIZE_MIB:.2f} MiB after retries; "
            f"last result was {human_mib(actual_size):.3f} MiB."
        )

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Compress a complete video to fit under Discord's 8 MiB limit."
    )
    parser.add_argument("filename", help="Path to the input video file")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic black-border crop detection")
    parser.add_argument("--no-audio", action="store_true", help="Remove audio and spend the whole bitrate budget on video")
    parser.add_argument(
        "--preset",
        default="slow",
        choices=["medium", "slow", "slower", "veryslow"],
        help="x264 preset (default: slow; slower gives slightly better quality but takes longer)",
    )
    parser.add_argument(
        "--bpp",
        type=float,
        default=DEFAULT_BPP,
        help=f"Target bits/pixel/frame used for automatic resolution selection (default: {DEFAULT_BPP})",
    )
    args = parser.parse_args()

    try:
        process_video(
            args.filename,
            no_crop=args.no_crop,
            no_audio=args.no_audio,
            preset=args.preset,
            target_bpp=max(0.005, args.bpp),
        )
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed.")
        if e.stderr:
            print(e.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()