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
parser.add_argument("-s", "--save", type=str, default="videos/results",
	help="path to save output file(s)")
args = parser.parse_args()

# hyperparameters
input_file = args.video
output_file = args.output
save_dir = args.save
temp_weight = 'best_temp.pt'
det_weight = 'yolov8x-pose.pt'
alpha = 0.5 # transparency parameter

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class YOLOv8_Temp:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, img):
        height, width, channels = img.shape

        result = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)[0]
        bboxes, class_ids, scores = [], [], []

        if result:
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
            scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        return bboxes, class_ids, scores

class YOLOv8_Model:
    def __init__(self, model_path, alpha=0.5, conf=0.3):
        self.model = YOLO(model_path)
        self.alpha = alpha
        self.conf = conf

    def detect(self, img):
        height, width, channels = img.shape

        result = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)[0]
        keypoints_idx = []
        bboxes, class_ids, scores = [], [], []

        if result:
            for k in result.keypoints:
                keypoints_idx.append(k)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
            scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        people_boxes, people_keypoints, people_score = [], [], []
        for bbox, class_id, kp, score in zip(bboxes, class_ids, keypoints_idx, scores):
            if class_id == 0:
                people_boxes.append(bbox)
                people_keypoints.append(kp)
                people_score.append(score)

        return people_boxes, people_keypoints, people_score


cap = cv2.VideoCapture(input_file)
model = YOLOv8_Model(det_weight)
model_temp = YOLOv8_Temp(temp_weight)

vid_writer = None

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_file = input_file.split('/')[-1][:-4] + '_result.mp4'

vid_writer = cv2.VideoWriter(os.path.join(save_dir, output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
it = 0


trash_assigner = 0
total_people = 0
current_people = 0
door_checker = False

door_color = {0: (0, 255, 0), 1: (255, 0, 0)}
person_text = 'person'

while True:
    print(it, '/', total_frames)
    it += 1

    ret, frame = cap.read()
    if not ret:
        break

    bboxes, class_ids, scores = model_temp.detect(frame)
    people_boxes, people_keypoints, people_score = model.detect(frame)

    if len(bboxes)>0:
        (dx, dy, dx2, dy2) = bboxes[0]

    if current_people != len(people_boxes) and door_checker:
        total_people += 1
        if total_people > 1:
            person_text = 'people'

    current_people = len(people_boxes)
    door_checker = False

    for bbox, kpts, s in zip(people_boxes, people_keypoints, people_score):
        (x, y, x2, y2) = bbox
        # kpt_line &= is_pose

        if dx < x < dx2 and dy < y < dy2:
            door_checker = True

        cv2.rectangle(frame, (x, y), (x2, y2), (0,255,255), 6)
        cv2.putText(frame, 'person', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 6, (0,255,255), 6)

    cv2.rectangle(frame, (dx, dy), (dx2, dy2), door_color[total_people%2], 6)
    cv2.putText(frame, 'target', (dx, dy-10), cv2.FONT_HERSHEY_PLAIN, 6, door_color[total_people%2], 6)
    cv2.putText(frame, str(total_people) + ' ' + person_text + ' entered', (80, 80), cv2.FONT_HERSHEY_PLAIN, 6, door_color[total_people%2], 6)
    # cv2.imshow('im', frame)
    frame = cv2.resize(frame, (width, height), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    vid_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
vid_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
