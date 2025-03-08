import cv2
import numpy as np
import subprocess
import os
import argparse

def auto_crop(image, threshold=10):
    """
    Determines the bounding box for non-black areas in a grayscale image.
    Pixels with a value above the threshold are considered non-black.
    Returns (x, y, width, height) of the detected crop region.
    """
    coords = np.where(image > threshold)
    if coords[0].size == 0 or coords[1].size == 0:
        # If all pixels are nearly black, return full image dimensions.
        return 0, 0, image.shape[1], image.shape[0]
    top = np.min(coords[0])
    bottom = np.max(coords[0])
    left = np.min(coords[1])
    right = np.max(coords[1])
    return left, top, right - left, bottom - top

def compute_common_crop(video_path, num_samples=100, threshold=10):
    """
    Analyzes random frames from the video and computes the common crop area
    by taking the intersection of the crop regions from each frame.
    Returns a tuple (x, y, width, height) representing the crop region.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.random.choice(total_frames, min(num_samples, total_frames), replace=False)
    crop_rects = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_rect = auto_crop(gray, threshold)
        crop_rects.append(crop_rect)
    cap.release()
    
    if not crop_rects:
        return None

    # Compute the intersection of all crop rectangles.
    x = max(rect[0] for rect in crop_rects)
    y = max(rect[1] for rect in crop_rects)
    right = min(rect[0] + rect[2] for rect in crop_rects)
    bottom = min(rect[1] + rect[3] for rect in crop_rects)
    new_width = right - x
    new_height = bottom - y
    if new_width <= 0 or new_height <= 0:
        return None
    return x, y, new_width, new_height

def compute_useful_frequency(image):
    """
    Performs a 2D Fourier transform on the image and computes a normalized cutoff frequency.
    It calculates the cumulative energy of the sorted spectral magnitudes and finds the index
    at which 90% of the total energy is reached.
    Returns a normalized value between 0 and 1, where 1 indicates high detail.
    """
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    sorted_magnitudes = np.sort(magnitude_spectrum.ravel())[::-1]
    energy = np.cumsum(sorted_magnitudes)
    total_energy = energy[-1]
    
    cutoff_index = np.where(energy >= 0.9 * total_energy)[0][0]
    normalized_frequency = cutoff_index / len(sorted_magnitudes)
    return normalized_frequency

def get_video_info(video_path):
    """
    Retrieves video information including width, height, FPS, and total frame count.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count

def compute_average_useful_detail(video_path, num_samples=100, crop_params=None):
    """
    Extracts a number of random frames from the video, optionally crops them according
    to crop_params, and computes the average normalized useful frequency.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.random.choice(total_frames, min(num_samples, total_frames), replace=False)
    
    useful_values = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if crop_params:
            x, y, w, h = crop_params
            gray = gray[y:y+h, x:x+w]
        useful_value = compute_useful_frequency(gray)
        useful_values.append(useful_value)
    
    cap.release()
    if not useful_values:
        raise ValueError("Failed to extract any frames for analysis.")
    return np.mean(useful_values)

def compute_optimal_resolution(video_path, crop_params=None):
    """
    Computes the optimal video resolution based on the average useful frequency detail
    from random frames. If crop_params is provided, it uses the cropped region's dimensions.
    The computed resolution respects Discord's maximum limits (1920x1080) and maintains aspect ratio.
    Returns optimal width, height, FPS, and total frame count.
    """
    orig_width, orig_height, fps, frame_count = get_video_info(video_path)
    if crop_params:
        # Use cropped dimensions.
        _, _, crop_width, crop_height = crop_params
        width, height = crop_width, crop_height
    else:
        width, height = orig_width, orig_height
    
    avg_detail = compute_average_useful_detail(video_path, num_samples=100, crop_params=crop_params)
    
    # Base width scaled by average detail (using 1980 as an arbitrary base factor).
    optimal_width = int(1980 * avg_detail)
    aspect_ratio = width / height
    optimal_height = int(optimal_width / aspect_ratio)
    
    # Enforce Discord's maximum resolution limits.
    if optimal_width > 1920:
        optimal_width = 1920
        optimal_height = int(optimal_width / aspect_ratio)
    if optimal_height > 1080:
        optimal_height = 1080
        optimal_width = int(optimal_height * aspect_ratio)
    
    # Adjust dimensions to be even numbers.
    if optimal_width % 2 != 0:
        optimal_width -= 1
    if optimal_height % 2 != 0:
        optimal_height -= 1
    
    return optimal_width, optimal_height, fps, frame_count

def generate_ffmpeg_command(input_file, output_file, width, height, fps, video_bitrate_kbps, crop_filter=None):
    """
    Generates the ffmpeg command for transcoding using the specified scaling, crop (if any),
    video bitrate, and enforces a maximum output file size of 8 MB.
    """
    # Build the video filter chain: apply crop first (if provided) then scale.
    if crop_filter:
        vf = f"{crop_filter},scale={width}:{height}"
    else:
        vf = f"scale={width}:{height}"
    
    command = [
        "ffmpeg",
        "-y",  # Automatically overwrite output file.
        "-i", input_file,
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "slow",
        "-b:v", f"{video_bitrate_kbps}k",
        "-maxrate", f"{video_bitrate_kbps}k",
        "-bufsize", f"{video_bitrate_kbps * 2}k",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-fs", "8000000",  # Enforce maximum file size of 8 MB.
        output_file
    ]
    return command

def process_video(input_file):
    """
    Main function:
      - Checks if the file exists.
      - Detects and computes common crop parameters to remove black borders.
      - Validates that the crop area is within the original video dimensions.
      - Determines the optimal resolution (after cropping) and calculates the required video bitrate
        so that the final file does not exceed 8 MB.
      - Transcodes the video using ffmpeg with crop and scale filters.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File {input_file} not found.")
    
    # Get original video dimensions.
    orig_width, orig_height, fps, frame_count = get_video_info(input_file)
    
    # Compute crop parameters to remove black borders (if any).
    crop_params = compute_common_crop(input_file, num_samples=100, threshold=10)
    if crop_params:
        x, y, w, h = crop_params
        # Validate crop region is within original dimensions.
        if x < 0 or y < 0 or (x + w) > orig_width or (y + h) > orig_height:
            print("Computed crop region is out of bounds. Skipping crop.")
            crop_params = None
    
    crop_filter = None
    if crop_params:
        # Build crop filter string: crop=width:height:x:y.
        crop_filter = f"crop={crop_params[2]}:{crop_params[3]}:{crop_params[0]}:{crop_params[1]}"
    
    optimal_width, optimal_height, fps, frame_count = compute_optimal_resolution(input_file, crop_params=crop_params)
    
    # Construct output filename: <original_filename>_discord.mp4.
    base_name, _ = os.path.splitext(input_file)
    output_file = f"{base_name}_discord.mp4"
    
    print(f"Optimal resolution: {optimal_width}x{optimal_height} @ {fps} FPS")
    
    # Calculate video duration in seconds.
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0:
        raise ValueError("Invalid video duration.")
    
    # Discord file size limit: 8 MB.
    max_size_bytes = 8 * 1024 * 1024  # 8 MB in bytes.
    max_size_bits = max_size_bytes * 8  # in bits.
    
    # Calculate target total bitrate (in bits per second).
    target_total_bitrate = max_size_bits / duration
    
    # Audio bitrate (128 kbps) in bits per second.
    audio_bitrate_bps = 128 * 1024
    # Calculate video bitrate (in bits per second) and convert to kbps.
    video_bitrate_bps = target_total_bitrate - audio_bitrate_bps
    # Enforce a minimum video bitrate for acceptable quality (e.g., 100 kbps).
    min_video_bitrate_bps = 100 * 1024
    if video_bitrate_bps < min_video_bitrate_bps:
        video_bitrate_bps = min_video_bitrate_bps
    video_bitrate_kbps = int(video_bitrate_bps / 1024)
    
    print(f"Target video bitrate: {video_bitrate_kbps} kbps (to keep final file under 8 MB)")
    print("Starting transcoding...")
    
    ffmpeg_command = generate_ffmpeg_command(
        input_file,
        output_file,
        optimal_width,
        optimal_height,
        fps,
        video_bitrate_kbps,
        crop_filter=crop_filter
    )
    
    print("Executing command:")
    print(" ".join(ffmpeg_command))
    
    try:
        result = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg error output:")
        print(e.stderr)
        raise e
    
    print(f"Transcoding complete! File saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized video transcoding for Discord (max file size: 8 MB)."
    )
    parser.add_argument("filename", help="Path to the input video file")
    
    args = parser.parse_args()
    try:
        process_video(args.filename)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
