from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import cv2
import numpy as np
import imutils
import time

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

# ##### TIME #####
# startTime = time.time()

IMAGE_FILE = 'image.jpg'
RESIZED_IMAGE_FILE = 'resized_image.jpg'

# MODEL_PATH = 'efficientdet.tflite'
# MODEL_PATH = 'lite-model_efficientdet_lite0_detection_metadata_1.tflite'
MODEL_PATH = 'mobilenetv2_ssd_256_uint8.tflite'

img = cv2.imread(IMAGE_FILE)

## RESIZE IMAGE
# # new_height = 128
# new_height = img.shape[0]
# resized_img = imutils.resize(img, height=new_height)
# cv2.imwrite(RESIZED_IMAGE_FILE, resized_img)

# ##### TIME #####
# executionTime = (time.time() - startTime)
# print('Load and resize image: ' + str(executionTime))
# startTime = time.time()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILE)

detection_result = detector.detect(image)
coord_list = detection_result.detections

# ##### TIME #####
# executionTime = (time.time() - startTime)
# print('Get Boxes: ' + str(executionTime))
# startTime = time.time()

all_xywh_list = []
counter = 0

for coord in coord_list:
    x_y_w_h = [coord.bounding_box.origin_x, coord.bounding_box.origin_y,
               coord.bounding_box.width, coord.bounding_box.height, coord.categories[0].category_name]
    all_xywh_list.append(x_y_w_h)
    cropped_image = img[x_y_w_h[1]:x_y_w_h[1] +
                                x_y_w_h[3], x_y_w_h[0]:x_y_w_h[0]+x_y_w_h[2]]
    cv2.imwrite("cropped_image_" +
                x_y_w_h[4] + "_" + str(counter) + ".jpg", cropped_image)
    counter += 1

print(all_xywh_list)

# image = cv2.imread(path, 0)
img_bw = img
# window_name = 'Image'
start_point = 0, 0
end_point = int(img_bw.shape[1]), int(img_bw.shape[0])
black_color = (0, 0, 0)
white_color = (255, 255, 255)
thickness = -1
img_bw = cv2.rectangle(img_bw, start_point, end_point, black_color, thickness)

for box in all_xywh_list:
    start_point = box[0], box[1]
    end_point = box[0]+box[2], box[1]+box[3]
    img_bw = cv2.rectangle(img_bw, start_point, end_point, white_color, thickness)

cv2.imwrite("black_and_white.jpg", img_bw)

# ##### TIME #####
# executionTime = (time.time() - startTime)
# print('Co-ordinates and images: ' + str(executionTime))
