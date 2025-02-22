import cv2
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip, AudioFileClip
import whisper
import os
import time

from moviepy.config import change_settings
change_settings({"FFMPEG_BINARY": "ffmpeg\\bin\\ffmpeg.exe"})

# Global variable to track processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'processed_file': None
}

def update_progress(progress):
    global processing_status
    processing_status['progress'] = progress

def crop_width(frame, detection_rect, target_aspect_ratio, current_crop_center, smoothing_factor=0.1):
    x, y, w, h = detection_rect
    center_x = x + w // 2
    frame_h, frame_w = frame.shape[:2]

    # Calculate crop width for 9:16 aspect ratio
    crop_h = frame_h
    crop_w = int(crop_h * target_aspect_ratio)
    
    if crop_w > frame_w:
        crop_w = frame_w

    # Define threshold for center movement
    movement_threshold = crop_w * 0.15  # 15% of crop width

    # Only update center if movement exceeds threshold
    if abs(center_x - current_crop_center) > movement_threshold:
        # Apply smoothing only when movement is necessary
        smoothing_factor = 0.3  # Stronger smoothing for transitions
        new_center = (1 - smoothing_factor) * current_crop_center + smoothing_factor * center_x
    else:
        # Keep the current center if movement is small
        new_center = current_crop_center

    # Ensure the crop window stays within frame bounds
    left = int(max(0, new_center - crop_w // 2))
    right = int(min(frame_w, left + crop_w))
    
    # Adjust if we hit frame boundaries
    if right == frame_w:
        left = frame_w - crop_w
    if left == 0:
        right = crop_w

    return frame[:, left:right], new_center

def detect_face_or_human(frame, face_cascade, hog):
    # First try face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Sort faces by size and pick the largest one
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        return faces[0], 'face'
    
    # If no face found, try human detection
    humans, weights = hog.detectMultiScale(
        frame, 
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        hitThreshold=0.3
    )
    
    if len(humans) > 0:
        # Sort humans by size and pick the largest one
        humans = sorted(humans, key=lambda x: x[2] * x[3], reverse=True)
        return humans[0], 'human'
    
    return None, None

def is_significant_movement(prev_rect, curr_rect, threshold=20):
    if prev_rect is None or curr_rect is None:
        return True
    
    prev_x, prev_y, prev_w, prev_h = prev_rect
    curr_x, curr_y, curr_w, curr_h = curr_rect
    
    prev_center_x = prev_x + prev_w // 2
    prev_center_y = prev_y + prev_h // 2
    curr_center_x = curr_x + curr_w // 2
    curr_center_y = curr_y + curr_h // 2
    
    dist = np.sqrt((prev_center_x - curr_center_x) ** 2 + (prev_center_y - curr_center_y) ** 2)
    
    # Add exponential moving average for distance
    if not hasattr(is_significant_movement, 'avg_dist'):
        is_significant_movement.avg_dist = dist
    else:
        is_significant_movement.avg_dist = 0.9 * is_significant_movement.avg_dist + 0.1 * dist
    
    return is_significant_movement.avg_dist > threshold

def process_frame(frame, face_cascade, hog, previous_rect, current_crop_center, movement_threshold=15, smoothing_factor=0.05):
    # Try to detect face or human
    detection_rect, detection_type = detect_face_or_human(frame, face_cascade, hog)
    
    if detection_rect is not None:
        if previous_rect is None or is_significant_movement(previous_rect, detection_rect, movement_threshold):
            previous_rect = detection_rect
    elif previous_rect is None:
        # If no detection and no previous rect, use center of frame
        height, width = frame.shape[:2]
        previous_rect = (width // 4, height // 4, width // 2, height // 2)

    cropped_frame, current_crop_center = crop_width(
        frame, 
        previous_rect, 
        9/16, 
        current_crop_center, 
        smoothing_factor
    )
    
    return cropped_frame, previous_rect, current_crop_center

def add_caption_to_frame(frame, caption, frame_width, frame_height):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    margin = 20
    max_width = frame_width - 2 * margin

    font_scale = 1.0
    (text_width, text_height), _ = cv2.getTextSize(caption, font, font_scale, thickness)
    
    text_x = (frame_width - text_width) // 2
    text_y = frame_height - margin - text_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 10, text_y - 10), (text_x + text_width + 10, text_y + text_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, caption, (text_x, text_y + text_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def recognize_speech_whisper(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, word_timestamps=True)
    captions = []
    for segment in result['segments']:
        for word in segment['words']:
            captions.append({"word": word["word"], "timestamps": (word["start"], word["end"])})
    return captions

def process_frame_with_caption(get_frame, t, captions, frame_width, frame_height, face_cascade, hog, previous_face, current_crop_center, smoothing_factor, movement_threshold):
    frame = get_frame(t)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_bgr, previous_face, current_crop_center = process_frame(
        frame_bgr, 
        face_cascade, 
        hog,
        previous_face, 
        current_crop_center, 
        movement_threshold, 
        smoothing_factor
    )

    word_to_display = None
    for item in captions:
        if item['timestamps'][0] <= t <= item['timestamps'][1]:
            word_to_display = item['word']
            break

    caption = word_to_display if word_to_display else ""
    processed_frame = add_caption_to_frame(frame_bgr, caption, frame_bgr.shape[1], frame_bgr.shape[0])

    return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), previous_face, current_crop_center

def main(input_video_path, output_video_path, processing_status, max_duration=10):
    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    try:
        # Update progress: Video loading started
        processing_status['progress'] = 10
        clip = VideoFileClip(input_video_path).subclip(0, min(max_duration, VideoFileClip(input_video_path).duration))
        print(f"Video loaded successfully. Duration: {clip.duration} seconds")

        # Update progress: Video loaded
        processing_status['progress'] = 20

        frame_width, frame_height = clip.w, clip.h
        current_crop_center = frame_width // 2
        previous_face = None

        # Update progress: Extracting audio
        processing_status['progress'] = 30
        audio_file = "extracted_audio.wav"
        print(f"Extracting audio to: {audio_file}")
        clip.audio.write_audiofile(audio_file)
        
        # Update progress: Audio extracted
        processing_status['progress'] = 40

        print("Recognizing speech using Whisper...")
        # Update progress: Speech recognition started
        processing_status['progress'] = 50
        captions = recognize_speech_whisper(audio_file)
        print(f"Speech recognition completed. Captions: {len(captions)} words")

        # Update progress: Speech recognition completed
        processing_status['progress'] = 60

        def make_frame(t):
            nonlocal previous_face, current_crop_center
            frame, previous_face, current_crop_center = process_frame_with_caption(
                clip.get_frame, 
                t, 
                captions, 
                frame_width, 
                frame_height, 
                face_cascade,
                hog,
                previous_face, 
                current_crop_center,
                smoothing_factor=0.3,     # Higher smoothing for transitions
                movement_threshold=100     # Much higher threshold to reduce frequency of movements
            )
            return frame

        print("Creating output video clip...")
        # Update progress: Creating output video
        processing_status['progress'] = 70
        output_clip = VideoClip(make_frame, duration=clip.duration).set_audio(clip.audio)
        print(f"Writing output video to: {output_video_path}")
        # Update progress: Writing output video
        processing_status['progress'] = 80
        output_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=24)
        print("Output video saved successfully.")

        # Update progress: Output video saved
        processing_status['progress'] = 90

        os.remove(audio_file)
        print(f"Temporary audio file removed: {audio_file}")

        clip.close()
        print("Video processing completed.")

        # Update progress: Processing completed
        processing_status['progress'] = 100
    except Exception as e:
        print(f"Error during video processing: {e}")
        raise