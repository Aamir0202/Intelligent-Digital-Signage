import os
import random
import queue
import cv2 as cv
import multiprocessing

from statistics import mode

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


def video_stream(frame_queue, processed_frame_queue, stop_event):
    cap = cv.VideoCapture("https://192.168.100.3:8081/video")
    if not cap.isOpened():
        Logger.error("Error: Unable to access the video source")
        stop_event.set()
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while not stop_event.is_set():
        try:
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
                stop_event.set()
                break
        except queue.Full:
            Logger.warning("Frame queue is full, skipping frame")
            continue
        except Exception as e:
            Logger.error(f"Error in video stream: {e}")
            stop_event.set()

    cap.release()
    cv.destroyWindow("Live Stream")
    cv.destroyWindow("Processed Frame")


def age_gender_detection(frame_queue, processed_frame_queue, stop_event, ad_manager):
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
                ad = recommend_ad(gender, age_group)
                Logger.info(f"Predictions: {predictions}, Ad: {ad}")
                processed_frame_queue.put(processed_frame)

                # Safely update ad recommendation in ad_manager
                with ad_manager["lock"]:
                    ad_manager["ad"] = ad
                ad_playback(ad)
        except queue.Empty:
            continue
        except Exception as e:
            Logger.error(f"Error in age_gender_detection: {e}")
            stop_event.set()


def gaze_stream(stop_event, ad_manager):
    gaze_processor = GazeProcessor()
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        Logger.error("Error: Unable to access the video source")
        return

    fps = 4
    delay = int(1000 / fps)
    attention_trace = []

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                Logger.error("Error: Unable to read the frame")
                break

            processed_frame, label = gaze_processor.process_frame(frame)
            cv.imshow("Gaze Stream", processed_frame)

            attention_trace.append(label)
            if len(attention_trace) == 20:
                attention_status = mode(attention_trace)
                attention_trace = []

                with ad_manager["lock"]:
                    current_ad = ad_manager["ad"]

                # Decide whether to change ad based on interest
                if attention_status != "Interested":
                    Logger.info("User not interested, switching ad...")
                    playsound(SOUND_FILE)

        except Exception as e:
            Logger.error(f"Error in gaze stream: {e}")
            stop_event.set()

        if cv.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyWindow("Gaze Stream")


def play_video(video_path, ad_queue, ad):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        Logger.info(f"Failed to open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not ad_queue.empty():
            break

        frame = cv.putText(
            frame, ad, (16, 16), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 255), 1
        )
        cv.imshow("Ad Playback", frame)

        if cv.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyWindow("Ad Playback")


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

    Logger.info(f"Playing ads from: {ad_folder}")
    video = random.choice(video_files) if len(video_files) > 1 else video_files[0]
    video_path = os.path.join(ad_folder, video)
    Logger.info(f"Now playing: {video}")

    ad_queue = multiprocessing.Queue()
    play_video(video_path, ad_queue, ad)

    while True:
        if ad_queue.get() == "CHANGE_AD":
            new_video = (
                random.choice(video_files) if len(video_files) > 1 else video_files[0]
            )
            video_path = os.path.join(ad_folder, new_video)
            Logger.info(f"Switching to: {new_video}")
            play_video(video_path, ad_queue, ad)
            break


def main():
    Config.load()
    Logger.init()

    frame_queue = multiprocessing.Queue(maxsize=1)
    processed_frame_queue = multiprocessing.Queue(maxsize=1)
    stop_event = multiprocessing.Event()

    # Using a manager to safely share ad information between processes
    with multiprocessing.Manager() as manager:
        ad_manager = manager.dict()
        ad_manager["ad"] = None
        ad_manager["lock"] = manager.Lock()

        stream_process = multiprocessing.Process(
            target=video_stream, args=(frame_queue, processed_frame_queue, stop_event)
        )
        detection_process = multiprocessing.Process(
            target=age_gender_detection,
            args=(frame_queue, processed_frame_queue, stop_event, ad_manager),
        )
        gaze_process = multiprocessing.Process(
            target=gaze_stream, args=(stop_event, ad_manager)
        )

        try:
            stream_process.start()
            detection_process.start()
            gaze_process.start()

            stream_process.join()
            detection_process.join()
            gaze_process.join()
        except KeyboardInterrupt:
            Logger.info("Shutting down due to user interruption.")
            stop_event.set()

        finally:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
