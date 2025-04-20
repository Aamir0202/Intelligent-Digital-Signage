from statistics import median, mode

import cv2 as cv
import numpy as np

from idsense import IDSENSE_DIR
from idsense.utils.drawing import draw_rounded_rectangle, draw_text
from idsense.utils.resources import download_resources

FACE_PROTO = str(IDSENSE_DIR / "models/caffe/face.prototxt")
FACE_CAFFEMODEL = str(
    IDSENSE_DIR / "models/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

AGE_PROTO = str(IDSENSE_DIR / "models/imdb/age.prototxt")
AGE_CAFFEMODEL = str(IDSENSE_DIR / "models/imdb/dex_chalearn_iccv2015.caffemodel")

GENDER_PROTO = str(IDSENSE_DIR / "models/imdb/gender.prototxt")
GENDER_CAFFEMODEL = str(IDSENSE_DIR / "models/imdb/gender.caffemodel")


RESOURCES = {
    AGE_PROTO: "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt",
    AGE_CAFFEMODEL: "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel",
    GENDER_PROTO: "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt",
    GENDER_CAFFEMODEL: "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel",
    FACE_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    FACE_CAFFEMODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
}

OUTPUT_INDEXES = np.array([i for i in range(0, 101)])


class ModelManager:
    _m_face = None
    _m_age = None
    _m_gender = None

    @staticmethod
    def initialize():
        """Initialize the models if they are not already initialized."""

        download_resources(RESOURCES)

        ModelManager._m_face = cv.dnn.readNetFromCaffe(FACE_PROTO, FACE_CAFFEMODEL)
        ModelManager._m_age = cv.dnn.readNetFromCaffe(AGE_PROTO, AGE_CAFFEMODEL)
        ModelManager._m_gender = cv.dnn.readNetFromCaffe(
            GENDER_PROTO, GENDER_CAFFEMODEL
        )

    @staticmethod
    def get_face_model():
        return ModelManager._m_face

    @staticmethod
    def get_age_model():
        return ModelManager._m_age

    @staticmethod
    def get_gender_model():
        return ModelManager._m_gender


def detect_faces(frame, confidence_threshold=0.5):
    """Detect faces in the given frame."""

    face_model = ModelManager.get_face_model()

    blob = cv.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 177.0, 123.0))

    face_model.setInput(blob)
    output = np.squeeze(face_model.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            start_x, start_y, end_x, end_y = box.astype(int)
            faces.append(
                (max(start_x, 0), max(start_y, 0), max(end_x, 0), max(end_y, 0))
            )
    return faces


def analyze_face(face_img):
    """Predict age and gender for a single face."""

    age_model = ModelManager.get_age_model()
    gender_model = ModelManager.get_gender_model()

    img_blob = cv.dnn.blobFromImage(face_img, size=(224, 224))

    age_model.setInput(img_blob)
    age_dist = age_model.forward()[0]
    age = round(np.sum(age_dist * OUTPUT_INDEXES))

    gender_model.setInput(img_blob)
    gender_class = gender_model.forward()[0]
    gender = "woman" if np.argmax(gender_class) == 0 else "man"

    return age, gender


def predict_age_n_gender(frame, callback):
    """Process a single video frame for face detection and analysis, sending data via callback."""

    predictions = []
    faces = detect_faces(frame)
    for face in faces:
        start_x, start_y, end_x, end_y = face
        detected_face = frame[start_y:end_y, start_x:end_x]
        age, gender = analyze_face(detected_face)
        predictions.append((age, gender, face))
        if callback:
            frame = callback(frame, age, gender, face)
    return frame, predictions


def annotate_age_n_gender(frame, age, gender, loc):
    """Annotates the given frame with age and gender information at the specified position."""

    start_x, start_y, end_x, end_y = loc
    label = f"{gender}, {age} years".title()

    overlay = frame.copy()
    draw_rounded_rectangle(
        overlay, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=-1
    )
    frame = cv.addWeighted(overlay, 0.3, frame, 0.7, 0)

    frame = draw_rounded_rectangle(frame, (start_x, start_y), (end_x, end_y))
    draw_text(frame, label, (start_x + 3, start_y - 4))

    return frame


def aggregate_age_n_gender(predictions):
    """Aggregate age and gender from multiple predictions efficiently."""

    if not predictions:
        return None, None
    ages, genders = zip(*[(age, gender) for age, gender, _ in predictions])
    try:
        majority_gender = mode(genders)
    except:
        majority_gender = genders[0]
    median_age = median(ages)
    return median_age, majority_gender
