import cv2
import numpy as np
import math
from keras.models import load_model
from typing import List, Tuple, Union
import mediapipe as mp
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# model = load_model("D:/django projects/faceMaskDetectionProject/media/model-faceMask_valacc_9948.h5")
model = load_model(
    "D:\django projects\FaceMaskDetection\media\model-faceMask_valacc_9948.h5")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

outputs = {0: 'Mask', 1: 'No mask'}
GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

rect_size = 4


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return 0, 0
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

    def get_frame(self):
        success, image = self.video.read()

        with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
            else:
                try:

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)

                    # Draw the face detection annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detections:
                        for detection in results.detections:

                            image_rows, image_cols, _ = image.shape

                            x = detection.location_data.relative_bounding_box.xmin
                            y = detection.location_data.relative_bounding_box.ymin
                            w2 = detection.location_data.relative_bounding_box.width
                            h2 = detection.location_data.relative_bounding_box.height

                            x1, y1 = _normalized_to_pixel_coordinates(
                                x, y, image_cols, image_rows)
                            x2, y2 = _normalized_to_pixel_coordinates(
                                x + w2,
                                y + h2, image_cols,
                                image_rows)
                            # print(int(y*100), int((y + h2)*100), int(x*100), int((x + w2)*100))
                            # print(y, y - h2, x, x + w2)
                            face_img = image[y1: y2+18, x1: x2]
                            face_img = cv2.resize(face_img, (150, 150))
                            face_img = face_img/255.0
                            face_img = np.reshape(face_img, (1, 150, 150, 3))
                            face_img = np.vstack([face_img])
                            result = model.predict(face_img)

                            label = np.argmax(result, axis=1)[0]
                            print(result, label, x1, y2)
                            cv2.putText(image, outputs[label] + " " + str(round(result[0][label]*100, 2)) + "%",
                                        (x1, y2 + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, GR_dict[label], 1)
                            cv2.rectangle(image, (x1, y1-20),
                                          (x2, y2+10), GR_dict[label], 2)
                except:
                    pass
        ret, jpeg = cv2.imencode(".jpg", image)
        return jpeg.tobytes()
