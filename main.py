import os
import random
import time
import queue
from statistics import mode
from collections import deque


import cv2 as cv
import multiprocessing

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

SOUND_FILE = "resources/audio/fresh-drop.mp3"

manager = multiprocessing.Manager()
ad = manager.Value("u", "", lock=True)
age_gender_info = manager.dict()
change_ad_flag = multiprocessing.Value("b", False)
recent_ads = deque(maxlen=3)
interested_start_times = manager.dict()
interested_durations = manager.dict()


def video_stream(frame_queue, processed_frame_queue, stop_event):
    Logger.info("Starting video stream...")
    cap = cv.VideoCapture("https://192.168.100.3:8081/video")
    if not cap.isOpened():
        Logger.error("Error: Unable to access the video source")
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
    frame_queue, processed_frame_queue, stop_event, ad, age_gender_info
):
    Logger.info("Initializing Age and Gender Detection...")
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
                processed_frame_queue.put(processed_frame)
                ad_playback(ad.value)
        except queue.Empty:
            Logger.warning("Age/Gender Detection: Frame queue was empty.")
            continue


def gaze_stream(stop_event, change_ad_flag, ad, age_gender_info):
    Logger.info("Gaze Stream initialized.")
    gaze_processor = GazeProcessor()
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        Logger.error("Error: Unable to access the gaze camera")
        return

    observation_period = 5
    fps = 8
    window_size = fps * observation_period
    delay = int(1000 / fps)
    # not_interested_start = None
    # interested_start = None
    attention_trace = []
    attention_status = "Interested"

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            Logger.error("Error: Unable to read frame from gaze camera")
            break

        processed_frame, label = gaze_processor.process_frame(frame)
        cv.imshow("Gaze Stream", processed_frame)

        attention_trace.append(label)
        if len(attention_trace) == window_size:
            attention_status = mode(attention_trace)
            attention_trace = []
            Logger.info(f"Detected Gaze Attention: {attention_status}")

        # person_detected = label != "No Person Detected"
        # current_time = time.time()

        if attention_status == "Not Interested":
            playsound(SOUND_FILE)
            with change_ad_flag.get_lock():
                change_ad_flag.value = True
        elif attention_status == "No Person Detected":
            with change_ad_flag.get_lock():
                change_ad_flag.value = True

        # if person_detected:
        #     if label == "Not Interested":
        #         if not_interested_start is None:
        #             not_interested_start = current_time
        #         elif current_time - not_interested_start > 5:
        #             if interested_start is not None:
        #                 duration = current_time - interested_start
        #                 Logger.info(
        #                     f"Viewer was interested in '{ad.value}' for {duration:.2f} seconds | Age: {age_gender_info.get('age')} | Gender: {age_gender_info.get('gender')}"
        #                 )
        #                 interested_start_times[ad.value] = duration
        #                 interested_durations[ad.value] = duration
        #                 interested_start = None

        #             Logger.info(
        #                 "User not interested for over 5 seconds. Triggering ad switch..."
        #             )
        #             playsound(SOUND_FILE)
        #             with change_ad_flag.get_lock():
        #                 change_ad_flag.value = True
        #             not_interested_start = None
        #     else:
        #         if interested_start is None:
        #             interested_start = current_time
        #         not_interested_start = None
        # else:
        #     not_interested_start = None
        #     interested_start = None

        if cv.waitKey(delay) & 0xFF == ord("q"):
            Logger.info("Gaze stream manually stopped.")
            break

    cap.release()
    cv.destroyWindow("Gaze Stream")
    Logger.info("Gaze Stream terminated.")


def play_video(video_path, video_name, ad, change_ad_flag):
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


def ad_playback(ad):
    ad_folder = os.path.join("ad_content", ad)

    if not os.path.exists(ad_folder) or not os.path.isdir(ad_folder):
        Logger.info(f"Ad directory not found: {ad_folder}")
        return

    video_files = [
        f for f in os.listdir(ad_folder) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        Logger.info(f"No videos found in {ad_folder}")
        return

    filtered_videos = [v for v in video_files if v not in recent_ads]

    if not filtered_videos:
        Logger.info("All videos recently played. Resetting history.")
        recent_ads.clear()
        filtered_videos = video_files

    video = random.choice(filtered_videos)
    recent_ads.append(video)

    video_path = os.path.join(ad_folder, video)
    Logger.info(f"Now playing: {video}")

    play_video(video_path, video, ad, change_ad_flag)

    with change_ad_flag.get_lock():
        change_ad_flag.value = False


def main():
    Config.load()
    Logger.init()

    Logger.info("Starting Intelligent Ad Display System...")
    frame_queue = multiprocessing.Queue(maxsize=1)
    processed_frame_queue = multiprocessing.Queue(maxsize=1)
    stop_event = multiprocessing.Event()

    stream_process = multiprocessing.Process(
        target=video_stream, args=(frame_queue, processed_frame_queue, stop_event)
    )
    detection_process = multiprocessing.Process(
        target=age_gender_detection,
        args=(frame_queue, processed_frame_queue, stop_event, ad, age_gender_info),
    )
    gaze_process = multiprocessing.Process(
        target=gaze_stream, args=(stop_event, change_ad_flag, ad, age_gender_info)
    )

    stream_process.start()
    detection_process.start()
    gaze_process.start()

    stream_process.join()
    detection_process.join()
    gaze_process.join()

    Logger.info("System shutting down. Cleaning up...")
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()


