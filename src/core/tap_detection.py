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

# === Tap Detection ===

def detect_taps(audio_data, sample_rate, threshold=0.04, frame_length=2048, hop_length=512, min_interval=0.2):
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.where(rms > threshold)[0]
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

    debounced_times = []
    debounced_frames = []
    last_tap_time = -min_interval
    for i, t in enumerate(times):
        if (t - last_tap_time) >= min_interval:
            debounced_times.append(t)
            debounced_frames.append(frames[i])
            last_tap_time = t

    return np.array(debounced_times), np.array(debounced_frames), rms


def plot_waveform_with_taps(audio_data, sr, tap_times, rms, threshold, frame_length=2048, hop_length=512):
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 6))
    librosa.display.waveshow(audio_data, sr=sr, alpha=0.5)
    plt.plot(times, rms, color='r', label='RMS')
    plt.axhline(y=threshold, color='g', linestyle='--')
    for t in tap_times:
        plt.axvline(x=t, color='purple', linestyle=':', alpha=0.5)
    plt.title("Detected Taps on Waveform")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Audio from Video ===

def extract_audio_from_video(video_path, output_wav=None):
    """
    Extracts audio from a video file and saves it as a .wav file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    print(f"[INFO] Extracting audio from: {video_path}")
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_wav)
    print(f"[SUCCESS] Audio saved to: {output_wav}")
    return output_wav


# === Tap Extraction and Saving ===

def extract_and_save_taps(audio_data, sr, tap_times, output_dir, material_type="unknown", tap_window=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for i, t in enumerate(tap_times):
        start = int((t - tap_window/2) * sr)
        end = int((t + tap_window/2) * sr)
        segment = audio_data[max(0, start):min(len(audio_data), end)]
        out_path = os.path.join(output_dir, f"{material_type}_tap_{i+1}.wav")
        sf.write(out_path, segment, sr)

# === Tap Conversion to NPY ===

import os
import numpy as np
import librosa

CONFIG = {
    "sampling_rate": 22050,
    "n_mels": 128,
    "fmax": 8000,
    "pad_or_truncate_len": 128 * 32,
    "valid_extensions": [".wav"]
}

def convert_taps_to_npy(audio_dir, tap_times, output_dir="temp_npy"):
    """
    Converts WAV files corresponding to detected taps into flattened Mel spectrogram .npy files.

    Args:
        audio_dir (str): Directory where per-tap WAV files are stored.
        tap_times (List[float]): List of tap times in seconds (used only for naming).
        output_dir (str): Directory to save the .npy files.
    
    Returns:
        List[str]: Paths to saved .npy feature files.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_paths = []

    for idx, _ in enumerate(tap_times):
        wav_path = os.path.join(audio_dir, f"test_tap_{idx+1}.wav")
        npy_path = os.path.join(output_dir, f"test_tap_{idx+1}.npy")
        try:
            mel = process_audio_file(wav_path)
            np.save(npy_path, mel)
            npy_paths.append(npy_path)
            print(f"✅ Saved NPY: {npy_path} | Shape: {mel.shape}")
        except Exception as e:
            print(f"❌ Failed for {wav_path}: {e}")
    
    return npy_paths

def process_audio_file(filepath: str) -> np.ndarray:
    y, sr = librosa.load(filepath, sr=CONFIG["sampling_rate"])
    return extract_melspectrogram(y, sr)

def extract_melspectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=CONFIG["n_mels"],
        fmax=CONFIG["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return pad_or_truncate(mel_db.flatten())

def pad_or_truncate(vector: np.ndarray) -> np.ndarray:
    target_len = CONFIG["pad_or_truncate_len"]
    current_len = len(vector)
    if current_len > target_len:
        return vector[:target_len]
    elif current_len < target_len:
        return np.pad(vector, (0, target_len - current_len))
    return vector


# === Tap Classification ===

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


# === Frame Extraction ===



import cv2
import os


def extract_frames(video_path, tap_times, output_dir, prefix="tap", frame_offset=0.08):
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
    os.makedirs(output_dir, exist_ok=True)

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



import re
import matplotlib.image as mpimg

def extract_numeric_index(path):
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
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import math

    total_images = len(image_paths)
    rows = math.ceil(total_images / images_per_row)

    figsize = (images_per_row * scale, rows * scale)  # Dynamic figure size
    fig, axs = plt.subplots(rows, images_per_row, figsize=figsize)
    axs = axs.flatten() if total_images > 1 else [axs]

    for ax in axs:
        ax.axis("off")

    for idx, path in enumerate(image_paths):
        img = mpimg.imread(path)
        axs[idx].imshow(img)
        axs[idx].set_title(f"Tap {idx + 1}")
        axs[idx].axis("off")

    plt.tight_layout()
    plt.show()


# === Call this after extracting frames ===




# === Hammer Tip Annotation Placeholder ===

import cv2
import numpy as np
import os

def annotate_hammer_tips(
    image_paths,
    output_dir=None,
    hsv_lower=(20, 100, 100),
    hsv_upper=(30, 255, 255),
    draw_radius=20
):
    """
    Detects and annotates the yellow hammer tip in each image using HSV filtering.

    Args:
        image_paths (List[str]): List of input frame paths.
        output_dir (str): Directory to save annotated frames (defaults to original folder).
        hsv_lower (tuple): Lower HSV bound for yellow detection.
        hsv_upper (tuple): Upper HSV bound for yellow detection.
        draw_radius (int): Radius of the circle to draw.

    Returns:
        Tuple[List[str], List[Tuple[int, int]]]: Paths to annotated images and detected (x, y) coordinates.
    """
    annotated_paths = []
    tap_coordinates = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[!] Could not read image: {path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))

            cv2.circle(image, center, draw_radius, (0, 0, 0), 20)
            #cv2.putText(image, "Hammer Tip", (center[0] + 10, center[1]),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            tap_coordinates.append(center)
            detected = True
            print(f"[✓] Annotated hammer tip at {center} in {os.path.basename(path)}")

        if not detected:
            tap_coordinates.append(None)
            print(f"[!] No hammer tip detected in {os.path.basename(path)}")

        # Save image
        save_path = os.path.join(output_dir or os.path.dirname(path), os.path.basename(path))
        cv2.imwrite(save_path, image)
        annotated_paths.append(save_path)

    return annotated_paths, tap_coordinates

# === Final Result Visualization ===

def visualize_taps_on_frame(frame_path, annotations, predictions, output_path="final_result.jpg"):
    frame = cv2.imread(frame_path)
    for i, (path, (x, y)) in enumerate(annotations.items()):
        color = (0, 255, 0) if predictions[i] == "good" else (0, 0, 255)
        cv2.circle(frame, (x, y), 15, color, 3)
    cv2.imwrite(output_path, frame)

import csv

def load_annotations(csv_path):
    annotations = {}
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = row["image"].strip()
            x = int(row["tap_x"])
            y = int(row["tap_y"])
            annotations[image_path] = (x, y)
    return annotations

import cv2
from PIL import Image as PILImage
from IPython.display import Image, display


def visualize_taps_on_frame(frame_path, annotations, predictions, output_path="final_result.jpg"):
    """
    Draws colored circles on the last frame to indicate tap classification results.

    Args:
        frame_path (str): Path to the frame image.
        annotations (dict): {image_path: (x, y)}
        predictions (list): [(filename, probability, label)]
        output_path (str): Path to save the final annotated image.
    """
    frame = cv2.imread(frame_path)

    # Ensure alignment between annotations and predictions
    for i, ((img_path, (x, y)), (_, prob, label)) in enumerate(zip(annotations.items(), predictions)):
        color = (0, 255, 0) if label == 'good' else (0, 0, 255)  # green or red
        radius = 10
        thickness = -1  # fill the circle
        cv2.circle(frame, (x, y), radius, color, thickness)
        # Optional: Show probability
        
        # === Add Legend with Larger Text ===
        legend_x, legend_y = 20, 20  # Top-left corner
        legend_w, legend_h = 200, 80  # Make box larger for bigger text
        overlay = frame.copy()

        # Semi-transparent white background
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Bigger font
        font_scale = 0.8
        thickness = 2

        # Green dot + label
        cv2.circle(frame, (legend_x + 20, legend_y + 25), 10, (0, 255, 0), -1)
        cv2.putText(frame, "Good tap", (legend_x + 40, legend_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Red dot + label
        cv2.circle(frame, (legend_x + 20, legend_y + 60), 10, (0, 0, 255), -1)
        cv2.putText(frame, "Bad tap", (legend_x + 40, legend_y + 67),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


    cv2.imwrite(output_path, frame)
    display(Image(filename=output_path))

import cv2
import os

def extract_last_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    # Total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"✅ Last frame saved to: {output_path}")
    else:
        print("❌ Failed to read the last frame.")
    
    cap.release()

    return output_path
