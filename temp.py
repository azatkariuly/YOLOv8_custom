from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np
import cv2
import argparse
import os

class_parameters = {0: ['rim', (0,0,255)], 1: ['backboard', (0,255,0)], 2: ['ball', (255,0,0)], 3: ['score', (255,255,0)]}

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", type=str, default="../videos/video_cut.mp4",
	help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="output.mp4",
	help="output file name")
parser.add_argument("-s", "--save", type=str, default="results",
	help="path to save output file(s)")
args = parser.parse_args()

# hyperparameters
input_file = args.video
output_video_path = args.output
save_dir = args.save
det_weight = 'best.pt'
# seg_weight = 'yolov8m-seg.pt'
person_weight = 'yolov8m.pt'
alpha = 0.5 # transparency parameter

class YOLOv8_Detection:
    def __init__(self, model_path, class_parameters, conf=0.5):
        self.model = YOLO(model_path, task='predict')
        self.cl_pr = class_parameters
        self.conf = conf

    def detect(self, img):
        height, width, channels = img.shape

        result = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)[0]

        # #later might be needed for segmentation
        # segmentation_contours_idx = []
        # if result.masks:
        #     for seg in result.masks.xyn:
        #         seg[:, 0] *= width
        #         seg[:, 1] *= height
        #         segment = np.array(seg, dtype=np.int32)
        #         segmentation_contours_idx.append(segment)

        bboxes, class_ids, scores = [], [], []
        ball_position = []
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
        class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
        scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        has_score = False

        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            (x, y, x2, y2) = bbox
            cv2.rectangle(img, (x,y), (x2,y2), self.cl_pr[class_id][1], 2)
            cv2.putText(img, self.cl_pr[class_id][0], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, self.cl_pr[class_id][1], 2)

            if class_id == 2:
                ball_position = bbox

            if class_id == 3:
                has_score = True

        return img, ball_position, has_score

class YOLOv8_Segmentation:
    def __init__(self, model_path, alpha=0.5, conf=0.35):
        self.model = YOLO(model_path)
        self.alpha = alpha
        self.conf = conf

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        bboxes, class_ids, scores = [], [], []

        if result:
            # for seg in result.masks.xyn:
            #     seg[:, 0] *= width
            #     seg[:, 1] *= height
            #     segment = np.array(seg, dtype=np.int32)
            #     segmentation_contours_idx.append(segment)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
            scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        people_boxes, people_seg, people_score = [], [], []
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            if class_id == 0:
                people_boxes.append(bbox)
                # people_seg.append(seg)
                people_score.append(score)

        return people_boxes, people_seg, people_score

        # return bboxes, class_ids, segmentation_contours_idx, scores

        # # segmentation
        # overlay = img.copy()
        # for bbox, class_id, seg, score in zip(bboxes, class_ids, segmentation_contours_idx, scores):
        #     if class_id == 0: # only people
        #         (x, y, x2, y2) = bbox
        #
        #         if len(ball_position)>0:
        #             if bbox[0] < ball_position[0] < bbox[2] and bbox[1] < ball_position[1] < bbox[3]:
        #                 cv2.fillPoly(overlay, [seg], (255,0,255))
        #
        # return cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        # return img

def detect_last_person(people, ball):
    if len(ball) > 0:
        for person in people:
            if (person[0] < ball[0] < person[2] and person[1] < ball[1] < person[3]) or (person[0] < ball[2] < person[2] and person[1] < ball[3] < person[3]):
                return person

    return None

cap = cv2.VideoCapture(input_file)
model_det = YOLOv8_Detection(det_weight, class_parameters)
model_seg = YOLOv8_Segmentation(person_weight)

frame_queue = []
vid_writer = None
vid_writer_forward = 0
vid_writer_counter = 0

seconds_forward = 2
seconds_back = 2

score_validation = 0
score_detection_threshold = 5

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
it = 0

last_person = None

while True:
    print(it, '/', total_frames)
    it += 1

    ret, frame = cap.read()
    if not ret:
        break

    # frame, ball_position, has_score = model_det.detect(frame)
    people_boxes, people_seg, people_score = model_seg.detect(frame)

    for bbox in people_boxes:
        (x1, y1, x2, y2) = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('image', frame)
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

if vid_writer is not None:
	vid_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
