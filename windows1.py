import os
import random
import time
import queue
from statistics import mode
from collections import deque, Counter
import gradio as gr

import cv2 as cv
import multiprocessing

import pygame

from playsound import playsound

from idsense.external.LaserGaze.GazeProcessor import GazeProcessor
from idsense.models.imdb import (
    ModelManager,
    predict_age_n_gender,
    annotate_age_n_gender,
    aggregate_age_n_gender,
)
from idsense.utils.demography import get_age_group
from idsense.utils.advertisment import recommend_ad
from idsense.utils import Config, Logger

SOUND_FILE = os.path.abspath("resources/audio/fresh-drop.mp3")
VIDEO_IP = None

from multiprocessing import Process, Manager

def start_windows(shared_data):
    global VIDEO_IP
    VIDEO_IP = shared_data.get('ip')
    print(f"windows1.py received IP: {VIDEO_IP}")
    main(shared_data)


# manager = multiprocessing.Manager()
# ad = manager.Value("u", "", lock=True)
# age_gender_info = manager.dict()
# change_ad_flag = multiprocessing.Value("b", False)
# recent_ads = deque(maxlen=3)
# interested_start_times = manager.dict()
# interested_durations = manager.dict()


def video_stream(frame_queue, processed_frame_queue, stop_event, shared_data):

    Config.load()  # Load configuration in the child process
    Logger.init()  # Initialize Logger in the child process
    Logger.info("Starting video stream...")
    print("Starting video stream...")
    ip = shared_data.get('ip', None)
    print(f"VIDEO STREAM received IP: {ip}")
    if not ip:
        Logger.error("No IP address provided for video stream.")
        stop_event.set()
        return

    cap = cv.VideoCapture(f"http://{ip}/video")

    if not cap.isOpened():
        Logger.error("Error: Unable to access the video source")
        print("Error: Unable to access the video source in VIDEO STREAM")
        stop_event.set()
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 25

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            Logger.error("Error: Unable to read the frame")
            break

        if not frame_queue.full():
            frame_queue.put(frame)
            cv.imshow("Live Stream", frame)

        if not processed_frame_queue.empty():
            processed_frame = processed_frame_queue.get()
            cv.imshow("Processed Frame", processed_frame)

        if cv.waitKey(delay) & 0xFF == ord("q"):
            Logger.info("Stream manually stopped by user")
            stop_event.set()
            break

    cap.release()
    cv.destroyWindow("Live Stream")
    cv.destroyWindow("Processed Frame")
    Logger.info("Video stream terminated.")


def age_gender_detection(
    frame_queue, processed_frame_queue, stop_event, ad, age_gender_info, recent_ads, change_ad_flag, interested_start_times, interested_durations, shared_video_path
):
    Config.load()  # Load configuration in the child process
    Logger.init()  # Initialize Logger in the child process
    Logger.info("Initializing Age and Gender Detection...")
    print("Initializing Age and Gender Detection...")
    ModelManager.initialize()
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            processed_frame, predictions = predict_age_n_gender(
                frame, annotate_age_n_gender
            )
            if predictions:
                age, gender = aggregate_age_n_gender(predictions)
                age_group = get_age_group(age)
                age_gender_info["age"] = age
                age_gender_info["gender"] = gender
                age_gender_info["age_group"] = age_group
                ad.value = recommend_ad(gender, age_group)
                Logger.info(f"Predictions: {predictions} | Recommended Ad: {ad.value}")
                print(f"Predictions: {predictions} | Recommended Ad: {ad.value}")
                processed_frame_queue.put(processed_frame)
                ad_playback(ad.value, recent_ads, change_ad_flag, interested_start_times, interested_durations,shared_video_path)
        except queue.Empty:
            Logger.warning("Age/Gender Detection: Frame queue was empty.")
            continue


def gaze_stream(stop_event, change_ad_flag, ad, age_gender_info):
    Config.load()  # Load configuration in the child process
    Logger.init()  # Initialize Logger in the child process
    Logger.info("Gaze Stream initialized.")
    gaze_processor = GazeProcessor()
    cap = cv.VideoCapture(0)  # Ensure the correct camera index is used
    if not cap.isOpened():
        Logger.error("Error: Unable to access the gaze camera")
        return

    observation_period = 5
    fps = 8
    window_size = fps * observation_period
    delay = int(1000 / fps)
    attention_trace = []
    attention_status = "Interested"
    cooldown = False  # Cooldown flag to prevent immediate ad changes

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            Logger.error("Error: Unable to read frame from gaze camera")
            break

        processed_frame, label = gaze_processor.process_frame(frame)
        cv.imshow("Gaze Stream", processed_frame)

        attention_trace.append(label)
        if len(attention_trace) == window_size:
            attention_status = weighted_attention_status(attention_trace)
            print("Attention Status:", attention_status)
            attention_trace = []
            Logger.info(f"Detected Gaze Attention: {attention_status}")

        if attention_status == "Not Interested" and not cooldown:
            play_sound(SOUND_FILE)
            with change_ad_flag.get_lock():
                change_ad_flag.value = True
            attention_trace = []  # Clear attention trace after ad switch
            cooldown = True  # Activate cooldown
            Logger.info("Cooldown activated after ad switch.")
        elif attention_status == "No Person Detected" and not cooldown:
            with change_ad_flag.get_lock():
                change_ad_flag.value = True
            attention_trace = []  # Clear attention trace after ad switch
            cooldown = True  # Activate cooldown
            Logger.info("Cooldown activated after ad switch.")

        # Reset the change_ad_flag and cooldown if attention is "Interested"
        if attention_status == "Interested":
            with change_ad_flag.get_lock():
                if change_ad_flag.value:
                    Logger.info("Resetting change_ad_flag as attention is now 'Interested'")
                change_ad_flag.value = False
            cooldown = False  # Deactivate cooldown

        if cv.waitKey(delay) & 0xFF == ord("q"):
            Logger.info("Gaze stream manually stopped.")
            stop_event.set()
            break

    cap.release()
    cv.destroyWindow("Gaze Stream")
    Logger.info("Gaze Stream terminated.")


def play_video(video_path, video_name, ad, change_ad_flag, interested_start_times, interested_durations):
    Logger.info(f"Loading video: {video_name}")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        Logger.info(f"Failed to open video: {video_name}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with change_ad_flag.get_lock():
            if change_ad_flag.value:
                Logger.info(
                    f"Video '{video_name}' interrupted due to lack of interest."
                )
                break

        frame = cv.putText(
            frame, ad, (16, 16), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1
        )

        cv.imshow("Ad Playback", frame)
        if cv.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyWindow("Ad Playback")

    if video_name in interested_start_times:
        Logger.info(
            f"Video '{video_name}' finished. Viewer was interested for {interested_start_times[video_name]:.2f} seconds | Age: {age_gender_info.get('age')} | Gender: {age_gender_info.get('gender')}"
        )
        del interested_start_times[video_name]
    elif video_name in interested_durations:
        Logger.info(
            f"Video '{video_name}' finished. Viewer was interested for {interested_durations[video_name]:.2f} seconds | Age: {age_gender_info.get('age')} | Gender: {age_gender_info.get('gender')}"
        )
        del interested_durations[video_name]


def ad_playback(ad, recent_ads, change_ad_flag, interested_start_times, interested_durations, shared_video_path):
    print("Starting ad playback...")
    ad_folder = os.path.join("ad_content", ad)

    if not os.path.exists(ad_folder) or not os.path.isdir(ad_folder):
        Logger.info(f"Ad directory not found: {ad_folder}")
        shared_video_path.value = None
        return

    video_files = [
        f for f in os.listdir(ad_folder) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        Logger.info(f"No videos found in {ad_folder}")
        shared_video_path.value = None
        return

    filtered_videos = [v for v in video_files if v not in recent_ads]

    if not filtered_videos:
        Logger.info("All videos recently played. Resetting history.")
        recent_ads.clear()
        filtered_videos = video_files

    video = random.choice(filtered_videos)
    recent_ads.append(video)

    video_path = os.path.join(ad_folder, video)
    Logger.info(f"Video Path: {video_path}")
    Logger.info(f"Now playing: {video}")

    # Update the shared variable with the video path
    shared_video_path.value = video_path
    print(f"windows1.py received video path........: {shared_video_path.value}")

    play_video(video_path, video, ad, change_ad_flag, interested_start_times, interested_durations)

    # Reset change_ad_flag after playing the ad
    with change_ad_flag.get_lock():
        Logger.info("Resetting change_ad_flag after ad playback")
        change_ad_flag.value = False


def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def weighted_attention_status(attention_trace):
    weights = [i + 1 for i in range(len(attention_trace))]  # Increasing weights
    weighted_counts = Counter()
    for label, weight in zip(attention_trace, weights):
        weighted_counts[label] += weight
    return max(weighted_counts, key=weighted_counts.get)  # Return label with max weight


def main(shared_data):
    Config.load()
    Logger.init()

    Logger.info("Starting Intelligent Ad Display System...")

    # Initialize multiprocessing Manager and shared variables
    manager = multiprocessing.Manager()
    ad = manager.Value("u", "", lock=True)
    age_gender_info = manager.dict()
    change_ad_flag = multiprocessing.Value("b", False)
    recent_ads = deque(maxlen=3)
    interested_start_times = manager.dict()
    interested_durations = manager.dict()
    shared_video_path = manager.Value("u", "", lock=True)  # shared video path!

    shared_data['video_path'] = shared_video_path  # make it accessible to app.py

    # Initialize queues and events
    frame_queue = multiprocessing.Queue(maxsize=1)
    processed_frame_queue = multiprocessing.Queue(maxsize=1)
    stop_event = multiprocessing.Event()

    # Create and start processes
    stream_process = multiprocessing.Process(
        target=video_stream, args=(frame_queue, processed_frame_queue, stop_event, shared_data)
    )
    detection_process = multiprocessing.Process(
        target=age_gender_detection,
        args=(frame_queue, processed_frame_queue, stop_event, ad, age_gender_info, recent_ads, change_ad_flag, interested_start_times, interested_durations, shared_video_path),
    )
    gaze_process = multiprocessing.Process(
        target=gaze_stream, args=(stop_event, change_ad_flag, ad, age_gender_info)
    )

    stream_process.start()
    detection_process.start()
    gaze_process.start()

    # Wait for processes to complete
    stream_process.join()
    detection_process.join()
    gaze_process.join()

    Logger.info("System shutting down. Cleaning up...")
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()



