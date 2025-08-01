# tap_detection.py

import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
import moviepy.editor as mp
import re
import matplotlib.image as mpimg
from PIL import Image as PILImage
import math
import csv
from IPython.display import Image, display

# =======================
# Constants
# =======================

# === Audio Processing Parameters Constants ===
DEFAULT_TAP_WINDOW = 0.1  # Seconds (tap duration window)
SAMPLING_RATE = 22050         # Standard sampling rate for audio processing
N_MELS = 128                  # Number of Mel filter banks for spectrogram generation
FMAX = 8000                   # Maximum frequency for Mel spectrogram
MEL_FLAT_LEN = N_MELS * 32    # Flattened length for input to classifier (padding target)

# === File Handling ===
VALID_EXTENSIONS = [".wav"]   # Supported audio file extensions

# === Tap Detection Parameters Constants ===
RMS_THRESHOLD = 0.04          # Energy threshold to consider a frame as a potential tap
FRAME_LENGTH = 2048           # Frame size used in RMS calculation for tap detection
HOP_LENGTH = 512              # Hop length for RMS calculation
MIN_TAP_INTERVAL = 0.2        # Minimum time (in seconds) between consecutive taps to prevent duplicates
FRAME_OFFSET = 0.08           # Time offset (in seconds) to align detected tap to hammer tip

# === Tap Extraction Parameters Constants ===
TAP_WINDOW_SEC = 0.1          # Duration (in seconds) around each tap to extract as WAV

# === Classification Parameters Constants ===
CLASSIFICATION_THRESHOLD = 0.5  # Probability threshold for binary classification

# === Hammer Tip Detection Constants ===
HAMMER_DRAW_RECT_WIDTH = 70   # Width of rectangle to mark hammer tip
HAMMER_DRAW_RECT_HEIGHT = 70  # Height of rectangle
HSV_LOWER_YELLOW = (20, 100, 100)  # Lower HSV bound for yellow detection
HSV_UPPER_YELLOW = (30, 255, 255)  # Upper HSV bound for yellow detection

# === Final Result Visualization Constants ===

CIRCLE_RADIUS = 10
CIRCLE_THICKNESS = -1  # Filled circle
GOOD_COLOR = (0, 255, 0)  # Green
BAD_COLOR = (0, 0, 255)   # Red

# === Legend Constants ===
LEGEND_POS = (20, 20)  # x, y top-left
LEGEND_SIZE = (200, 80)
LEGEND_BG_COLOR = (255, 255, 255)
LEGEND_ALPHA = 0.1
LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_FONT_SCALE = 0.8
LEGEND_FONT_COLOR = (0, 0, 0)
LEGEND_FONT_THICKNESS = 2

FRAME_PROP_COUNT = cv2.CAP_PROP_FRAME_COUNT # Property identifier for total frame count in a video
FRAME_POS = cv2.CAP_PROP_POS_FRAMES   # Property identifier to set the position of the current frame

CSV_IMAGE_COL = "image"     # Column name in CSV for image filename
CSV_TAP_X_COL = "tap_x"     # Column name for tap x-coordinate
CSV_TAP_Y_COL = "tap_y"     # Column name for tap y-coordinate

# =======================
# Utility
# =======================

def delete_file_if_exists(path):
    if path and os.path.isfile(path):
        os.remove(path)
        print(f"[INFO] Existing file deleted: {path}")

def recreate_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    else:
        os.makedirs(path)
    print(f"[INFO] Output directory ready: {path}")

# =======================
# Tap Detection
# =======================
def detect_taps(
    audio_data,
    sample_rate,
    threshold=RMS_THRESHOLD,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH,
    min_interval=MIN_TAP_INTERVAL
):
    """
    Detect taps in audio using RMS energy and debouncing.

    Args:
        audio_data (np.ndarray): Raw audio waveform.
        sample_rate (int): Sampling rate of the audio.
        threshold (float): RMS threshold for tap detection.
        frame_length (int): Frame size for RMS calculation.
        hop_length (int): Hop size for RMS calculation.
        min_interval (float): Minimum interval between taps (in seconds).

    Returns:
        np.ndarray: Times (in seconds) of debounced taps.
        np.ndarray: Frame indices corresponding to taps.
        np.ndarray: RMS energy values.
    """
    print("🔍 [TAP DETECT] Calculating RMS energy...")
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    over_threshold_frames = np.where(rms > threshold)[0]
    tap_times_raw = librosa.frames_to_time(over_threshold_frames, sr=sample_rate, hop_length=hop_length)

    print(f"📈 [TAP DETECT] Raw taps above threshold: {len(tap_times_raw)}")

    # Debounce to remove nearby taps
    debounced_times = []
    debounced_frames = []
    last_tap_time = -min_interval
    for i, t in enumerate(tap_times_raw):
        if (t - last_tap_time) >= min_interval:
            debounced_times.append(t)
            debounced_frames.append(over_threshold_frames[i])
            last_tap_time = t

    print(f"✅ [TAP DETECT] Final taps after debouncing: {len(debounced_times)}")
    return np.array(debounced_times), np.array(debounced_frames), rms


# =======================
# Plot with Taps
# =======================
def plot_waveform_with_taps(
    audio_data,
    sample_rate,
    tap_times,
    rms,
    threshold=RMS_THRESHOLD,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
):
    """
    Plot audio waveform with RMS and tap markers.

    Args:
        audio_data (np.ndarray): Raw waveform.
        sample_rate (int): Audio sample rate.
        tap_times (np.ndarray): Tap timestamps (in seconds).
        rms (np.ndarray): RMS energy over time.
        threshold (float): RMS threshold.
        frame_length (int): Frame size used in RMS.
        hop_length (int): Hop length used in RMS.
    """
    print("📊 [PLOT] Visualizing waveform with tap detections...")
    times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_length)

    plt.figure(figsize=(15, 6))
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=0.5, label='Waveform')
    plt.plot(times, rms, color='r', linewidth=1.5, label='RMS Energy')
    plt.axhline(y=threshold, color='g', linestyle='--', label='RMS Threshold')

    for t in tap_times:
        plt.axvline(x=t, color='purple', linestyle=':', alpha=0.7)

    plt.title("Detected Taps on Audio Waveform", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / RMS")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("✅ [PLOT] Done displaying tap detection.")


# =======================
# Audio from Video
# =======================
def extract_audio_from_video(video_path, output_wav):
    """
    Extract audio from video file and save as .wav.

    Args:
        video_path (str): Path to the input video.
        output_wav (str): Output path for extracted audio (.wav)

    Returns:
        str: Path to the saved WAV file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] Video not found: {video_path}")

    delete_file_if_exists(output_wav)

    print(f"[INFO] Extracting audio from video: {video_path}")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_wav, verbose=False, logger=None)

    print(f"[SUCCESS] Audio saved to: {output_wav}")
    return output_wav


# =======================
# Tap Audio Extraction
# =======================
def extract_and_save_taps(
    audio_data,
    sr,
    tap_times,
    output_dir,
    material_type="unknown",
    tap_window=DEFAULT_TAP_WINDOW
):
    """
    Extracts short audio clips around each tap and saves them as separate WAV files.

    Args:
        audio_data (np.ndarray): Full audio waveform.
        sr (int): Sampling rate.
        tap_times (list of float): List of tap times (in seconds).
        output_dir (str): Directory to save the extracted tap WAV files.
        material_type (str): Prefix label for file naming.
        tap_window (float): Duration of tap clip in seconds.
    """
    recreate_dir(output_dir)

    half_window = tap_window / 2
    for i, t in enumerate(tap_times):
        start = int((t - half_window) * sr)
        end = int((t + half_window) * sr)
        segment = audio_data[max(0, start):min(len(audio_data), end)]

        out_file = os.path.join(output_dir, f"{material_type}_tap_{i+1}.wav")
        sf.write(out_file, segment, sr)
        print(f"[SAVED] Tap {i+1} -> {out_file}")

# =======================
# Tap Conversion to NPY
# =======================



def convert_taps_to_npy(audio_dir, tap_times, output_dir="temp_npy"):
    """
    Converts WAV files of individual taps into flattened Mel spectrogram .npy files.

    Args:
        audio_dir (str): Directory containing per-tap WAV files.
        tap_times (List[float]): List of tap timestamps (used for naming only).
        output_dir (str): Where to save the .npy files.

    Returns:
        List[str]: List of .npy file paths saved.
    """
    print(f"\n🚀 [START] Converting tap WAVs in '{audio_dir}' to Mel spectrogram .npy files")
    print(f"🛠️  Output directory: {output_dir}")
    print(f"🔢 Number of taps to process: {len(tap_times)}")

    recreate_dir(output_dir)
    npy_paths = []

    for idx, _ in enumerate(tap_times):
        wav_name = f"test_tap_{idx+1}.wav"
        npy_name = f"test_tap_{idx+1}.npy"
        wav_path = os.path.join(audio_dir, wav_name)
        npy_path = os.path.join(output_dir, npy_name)

        print(f"\n📁 [PROCESSING] Tap {idx+1}: {wav_name}")

        if not os.path.exists(wav_path):
            print(f"❌ [SKIP] WAV file not found: {wav_path}")
            continue

        try:
            mel = process_audio_file(wav_path)
            np.save(npy_path, mel)
            npy_paths.append(npy_path)
            print(f"✅ [SAVED] {npy_name} | Shape: {mel.shape}")
        except Exception as e:
            print(f"🔥 [ERROR] Failed to convert {wav_name} → {e}")

    print(f"\n✅ [COMPLETE] Converted {len(npy_paths)}/{len(tap_times)} taps successfully.")
    return npy_paths


# === Supporting Methods ===

def process_audio_file(filepath: str) -> np.ndarray:
    """
    Loads a WAV file and returns its flattened Mel spectrogram.
    """
    y, sr = librosa.load(filepath, sr=SAMPLING_RATE)
    return extract_melspectrogram(y, sr)

def extract_melspectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Computes Mel spectrogram and converts it to dB scale.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return pad_or_truncate(mel_db.flatten())

def pad_or_truncate(vector: np.ndarray) -> np.ndarray:
    """
    Pads or truncates a feature vector to MEL_FLAT_LEN length.
    """
    if len(vector) > MEL_FLAT_LEN:
        return vector[:MEL_FLAT_LEN]
    return np.pad(vector, (0, MEL_FLAT_LEN - len(vector)))


# =======================
# Tap Classification
# =======================

def classify_tap_quality(npy_paths, model_path, threshold=0.5):
    """
    Classify tap quality based on NPY feature files and a trained model.

    Args:
        npy_paths (List[str]): Paths to .npy files (flattened Mel spectrograms).
        model_path (str): Path to the trained Keras model (.h5).
        threshold (float): Decision threshold for binary classification.

    Returns:
        List[Tuple[str, float, str]]: (filename, probability, predicted_label)
    """
    print(f"📦 Loading model from: {model_path}")
    model = load_model(model_path)
    print(f"✅ Model successfully loaded.")

    results = []
    print("\n🔍 Performing predictions on NPY files:")
    
    for path in npy_paths:
        try:
            x = np.load(path).reshape(1, -1)
            prob = model.predict(x, verbose=0)[0][0]
            label = "good" if prob > threshold else "bad"
            fname = os.path.basename(path)
            results.append((fname, prob, label))
            status = "🟢" if label == "good" else "🔴"
            print(f"{status} {fname} → Probability: {prob:.4f}, Predicted: {label}")
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    print(f"\n✅ Classification complete for {len(results)} samples.")
    return results


# =======================
# Frame Extraction
# =======================
def extract_frames(video_path, tap_times, output_dir, prefix="tap", frame_offset=FRAME_OFFSET):
    """
    Extracts and saves frames from a video at specified tap timestamps.

    Each extracted frame is saved as an image file and annotated with the
    actual frame number to help with tracking and debugging. This function
    supports a frame offset (in seconds) to adjust for any lag or motion blur
    between the audio-detected tap and the most visually clear frame.

    """
    
    # Get video FPS to calculate precise frame numbers
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not fps or fps == 0:
        raise ValueError("Unable to retrieve FPS from video.")

    print(f"[INFO] Video FPS: {fps:.2f}")

    # Dynamically calculate frame offset for 5 frames ahead (smoother visuals)
    frame_offset = 5 / fps
    print(f"[INFO] Using dynamic frame offset: {frame_offset:.4f} seconds")

    # Prepare output directory
    recreate_dir(output_dir)

    # Re-open video for frame extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    saved_paths = []

    for idx, tap_time in enumerate(tap_times):
        # Compute exact frame number with offset
        frame_number = int((tap_time + frame_offset) * fps)

        # Seek to the calculated frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            # Annotate frame with frame number
            annotated_frame = frame.copy()
            cv2.putText(
                annotated_frame,
                f"Frame #: {frame_number}",
                (20, 40),  # Position on the image
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,        # Font scale
                (0, 0, 255),# Red color
                2,          # Thickness
                cv2.LINE_AA
            )

            # Save annotated frame
            filename = os.path.join(output_dir, f"{prefix}_frame_{idx+1}.jpg")
            cv2.imwrite(filename, annotated_frame)
            saved_paths.append(filename)
            print(f"[✓] Saved frame {frame_number} as {filename}")
        else:
            print(f"[!] Failed to extract frame at {tap_time:.2f}s (frame {frame_number})")

    cap.release()
    return saved_paths

# =======================
# Display Image
# =======================

def extract_numeric_index(path: str) -> int:
    """
    Extracts a numeric index from an image filename (e.g., "_12.jpg").

    Args:
        path (str): File path or filename.

    Returns:
        int: Extracted numeric index, or -1 if no match found.
    """
    match = re.search(r"_(\d+)\.jpg$", path)
    return int(match.group(1)) if match else -1

def display_images_in_grid(image_paths, images_per_row=4, scale=4):
    """
    Displays images in a grid with adjustable scale.

    Args:
        image_paths (List[str]): Paths to images.
        images_per_row (int): Number of images per row.
        scale (float): Scaling factor per image.
    """
    if not image_paths:
        print("⚠️ [WARN] No image paths provided to display_images_in_grid.")
        return

    total_images = len(image_paths)
    rows = math.ceil(total_images / images_per_row)

    figsize = (images_per_row * scale, rows * scale)  # Dynamic figure size
    fig, axs = plt.subplots(rows, images_per_row, figsize=figsize)
    axs = axs.flatten() if total_images > 1 else [axs]

    # Turn off all axes first
    for ax in axs:
        ax.axis("off")

    for idx, path in enumerate(image_paths):
        print(f"🖼️ [INFO] Displaying image: {path}")
        try:
            img = mpimg.imread(path)
            axs[idx].imshow(img)
            axs[idx].set_title(f"Tap {idx + 1}")
        except Exception as e:
            print(f"❌ [ERROR] Could not load image {path}: {e}")

    plt.tight_layout()
    plt.show()

# =======================
# Hammer Tip Annotation
# =======================

def annotate_hammer_tips(
    image_paths,
    output_dir=None,
    hsv_lower=HSV_LOWER_YELLOW,
    hsv_upper=HSV_UPPER_YELLOW,
    rect_width=HAMMER_DRAW_RECT_WIDTH,
    rect_height=HAMMER_DRAW_RECT_HEIGHT,
    rect_color=(0, 0, 0),
    rect_thickness=3
):
    """
    Detects and annotates the yellow hammer tip in each image using HSV filtering.
    Instead of a circle, a filled rectangle is drawn for better visibility.

    Args:
        image_paths (List[str]): List of input frame paths.
        output_dir (str): Directory to save annotated frames (defaults to original folder).
        hsv_lower (tuple): Lower HSV bound for yellow detection.
        hsv_upper (tuple): Upper HSV bound for yellow detection.
        rect_width (int): Width of the rectangle to draw.
        rect_height (int): Height of the rectangle to draw.
        rect_color (tuple): BGR color of the rectangle (default: black).
        rect_thickness (int): Thickness (-1 for filled).

    Returns:
        Tuple[List[str], List[Tuple[int, int] or None]]:
            Paths to annotated images and detected (x, y) coordinates (or None if not found).
    """
    if not image_paths:
        print("⚠️ [WARN] No input images provided to annotate_hammer_tips.")
        return [], []

    if output_dir:
        print(f"📁 [INFO] Annotated images will be saved to: {output_dir}")
        recreate_dir(output_dir)

    annotated_paths = []
    tap_coordinates = []

    for path in image_paths:
        print(f"🖼️ [INFO] Processing image: {os.path.basename(path)}")
        image = cv2.imread(path)

        if image is None:
            print(f"❌ [ERROR] Could not read image: {path}")
            tap_coordinates.append(None)
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))

            # Draw rectangle instead of circle
            top_left = (center[0] - rect_width // 2, center[1] - rect_height // 2)
            bottom_right = (center[0] + rect_width // 2, center[1] + rect_height // 2)
            cv2.rectangle(image, top_left, bottom_right, rect_color, rect_thickness)

            tap_coordinates.append(center)
            detected = True

            print(f"✅ [DETECTED] Hammer tip at {center} in {os.path.basename(path)}")
        else:
            tap_coordinates.append(None)
            print(f"⚠️ [WARN] No hammer tip detected in {os.path.basename(path)}")

        save_path = os.path.join(output_dir or os.path.dirname(path), os.path.basename(path))
        cv2.imwrite(save_path, image)
        annotated_paths.append(save_path)
        print(f"💾 [SAVED] Annotated image saved to {save_path}\n")

    return annotated_paths, tap_coordinates


# =======================
# Final Result Visualization
# =======================

def load_annotations(csv_path):
    """
    Loads tap annotations from a CSV file.

    Each row in the CSV should contain:
        - 'image': filename of the image/frame
        - 'tap_x': x-coordinate of the tap
        - 'tap_y': y-coordinate of the tap

    Args:
        csv_path (str): Path to the annotation CSV file.

    Returns:
        dict: A dictionary mapping image filename to (x, y) coordinate.
              Format — { "frame001.jpg": (123, 456), ... }
    """
    annotations = {}
    print(f"📥 Loading annotations from: {csv_path}")
    
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            image_path = row[CSV_IMAGE_COL].strip()
            try:
                x = int(row[CSV_TAP_X_COL])
                y = int(row[CSV_TAP_Y_COL])
                annotations[image_path] = (x, y)
                print(f"✅ Annotation {i+1}: {image_path} → ({x}, {y})")
            except ValueError:
                print(f"⚠️ Skipping invalid coordinates at row {i+1}: {row}")
    
    print(f"🧾 Total annotations loaded: {len(annotations)}")
    return annotations

def visualize_taps_on_frame(frame_path, annotations, predictions, output_path="final_result.jpg"):
    """
    Draws filled colored circles on the last frame to indicate tap classification results.

    Args:
        frame_path (str): Path to the frame image.
        annotations (dict): {image_path: (x, y)}
        predictions (list): [(filename, probability, label)]
        output_path (str): Path to save the final annotated image.
    """
    print(f"[INFO] Loading frame from: {frame_path}")
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[ERROR] Could not load frame from: {frame_path}")
        return

    print("[INFO] Drawing tap predictions...")
    # Draw each tap annotation as solid circle
    for i, ((img_path, (x, y)), (_, prob, label)) in enumerate(zip(annotations.items(), predictions)):
        color = (0, 255, 0) if label == 'good' else (0, 0, 255)  # Green or Red
        radius = 10
        thickness = -1  # -1 means filled circle
        cv2.circle(frame, (x, y), radius, color, thickness)
        print(f"    → Tap {i+1}: ({x},{y}), Label: {label}, Prob: {prob:.2f}")

    print("[INFO] Drawing legend overlay...")
    # Draw solid background for legend (optional, not transparent)
    legend_x, legend_y = LEGEND_POS
    legend_w, legend_h = LEGEND_SIZE
    cv2.rectangle(frame, (legend_x, legend_y),
                  (legend_x + legend_w, legend_y + legend_h),
                  LEGEND_BG_COLOR, -1)  # filled background

    # Green (Good tap)
    cv2.circle(frame, (legend_x + 20, legend_y + 25), CIRCLE_RADIUS, GOOD_COLOR, -1)
    cv2.putText(frame, "Good tap", (legend_x + 40, legend_y + 32),
                LEGEND_FONT, LEGEND_FONT_SCALE, LEGEND_FONT_COLOR, LEGEND_FONT_THICKNESS)

    # Red (Bad tap)
    cv2.circle(frame, (legend_x + 20, legend_y + 60), CIRCLE_RADIUS, BAD_COLOR, -1)
    cv2.putText(frame, "Bad tap", (legend_x + 40, legend_y + 67),
                LEGEND_FONT, LEGEND_FONT_SCALE, LEGEND_FONT_COLOR, LEGEND_FONT_THICKNESS)

    # Save and display
    cv2.imwrite(output_path, frame)
    print(f"[SUCCESS] Final annotated frame saved to: {output_path}")
    display(Image(filename=output_path))


def extract_last_frame(video_path: str, output_path: str) -> str:
    """
    Extracts and saves the last frame from a video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the last frame image.

    Returns:
        str: Path to the saved frame image if successful, else None.
    """
    if not os.path.exists(video_path):
        print(f"❌ Video path does not exist: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video file: {video_path}")
        return None

    total_frames = int(cap.get(FRAME_PROP_COUNT))
    print(f"[INFO] Total frames in video: {total_frames}")

    if total_frames == 0:
        print("❌ Video has no frames.")
        cap.release()
        return None

    # Go to the last frame
    cap.set(FRAME_POS, total_frames - 1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read the last frame.")
        cap.release()
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"✅ Last frame saved to: {output_path}")

    cap.release()
    return output_path