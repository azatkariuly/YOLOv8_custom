from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np
import cv2
import argparse
import os

class_parameters = {0: ['rim', (0,0,255)], 1: ['backboard', (0,255,0)], 2: ['ball', (255,0,0)], 3: ['score', (255,255,0)]}

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v1", "--video1", type=str, default="../videos/video_cut.mp4",
	help="path to input video file")
parser.add_argument("-v2", "--video2", type=str, default="../videos/video_cut.mp4",
	help="path to input video file")
parser.add_argument("-o", "--output", type=str, default="output.mp4",
	help="output file name")
parser.add_argument("-s", "--save", type=str, default="results",
	help="path to save output file(s)")
args = parser.parse_args()

# hyperparameters
input_file1 = args.video1
input_file2 = args.video2

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

cap1 = cv2.VideoCapture(input_file1)
cap2 = cv2.VideoCapture(input_file2)

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

width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
it = 0

last_person1 = None
last_person2 = None

while True:
    print(it, '/', total_frames)
    it += 1

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    frame1, ball_position1, has_score1 = model_det.detect(frame1)
    people_boxes1, people_seg1, people_score1 = model_seg.detect(frame1)


    frame2, ball_position2, has_score2 = model_det.detect(frame2)
    people_boxes2, people_seg2, people_score2 = model_seg.detect(frame2)

    last_person_frame1 = detect_last_person(people_boxes1, ball_position1)
    last_person_frame2 = detect_last_person(people_boxes2, ball_position2)

    if last_person_frame1 is not None:
        (x, y, x2, y2) = last_person_frame1
        last_person1 = frame1[y:y2, x:x2]

        cv2.rectangle(frame1, (x,y), (x2,y2), (255,255,255), 2)

    if last_person_frame2 is not None:
        (x, y, x2, y2) = last_person_frame2
        last_person2 = frame2[y:y2, x:x2]

        cv2.rectangle(frame2, (x,y), (x2,y2), (255,255,255), 2)

    # if len(frame_queue) > fps*seconds_back:
    #     frame_queue.pop(0)
    # frame_queue.append(frame)
	#
    # if vid_writer_forward > 0:
    #     vid_writer_forward -= 1
    #     vid_writer.write(frame)
    #     continue
	#
    # if has_score:
    #     score_validation += 1
	#
    #     if score_validation >= score_detection_threshold and vid_writer_forward == 0:
    #         # save_path = str(Path(str(len(vid_writer) + 1)).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
	#
    #         if vid_writer is not None:
    #             vid_writer.release()
	#
    #         vid_writer_counter += 1
    #         vid_writer = cv2.VideoWriter(os.path.join(save_dir, str(vid_writer_counter) + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
	#
    #         cv2.imwrite(os.path.join(save_dir, str(vid_writer_counter) + '.jpg'), last_person)
	#
    #         # write from queue buffer
    #         for k in frame_queue:
    #             vid_writer.write(k)
	#
    #         vid_writer_forward = fps*seconds_forward
    # else:
    #     score_validation = 0
    # frame, ball_position = model_det.detect(frame)
    # frame = model_seg.detect(frame, ball_position)

    cv2.imshow('image1', frame1)
    cv2.imshow('image2', frame2)
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

if vid_writer is not None:
	vid_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
