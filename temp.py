from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np
import cv2

class_parameters = {0: ['rim', (0,0,255)], 1: ['backboard', (0,255,0)], 2: ['ball', (255,0,0)], 3: ['score', (255,255,0)]}

#straighforward algorithm
def find_closest(d, y, ball_position, new_seg):
    for i in range(len(d)):
        if len(y):
            idx = np.abs(y-d[i][0]).sum(axis=1).argmin()

            if len(ball_position) > 0:
                if (y[idx][0] < ball_position[0] < y[idx][2] and y[idx][1] < ball_position[1] < y[idx][3]) or (y[idx][0] < ball_position[2] < y[idx][2] and y[idx][1] < ball_position[3] < y[idx][3]):
                    d[i] = [y[idx], True, new_seg[idx]]
                else:
                    d[i] = [y[idx], False, new_seg[idx]]
            else:
                d[i] = [y[idx], False, new_seg[idx]]
            y = np.delete(y, idx, 0)
            new_seg = new_seg[:idx] + new_seg[idx+1:]
            # new_seg = np.delete(new_seg, idx, 0)

    return d

class YOLOv8_Detection:
    def __init__(self, model_path, class_parameters, conf=0.5):
        self.model = YOLO(model_path, task='predict')
        self.cl_pr = class_parameters
        self.conf = conf

    def has_score(self, img):
        result = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)[0]
        class_ids = np.array(result.boxes.cls.cpu(), dtype='int')

        if 3 in class_ids:
            return True
        return False

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

        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            (x, y, x2, y2) = bbox
            cv2.rectangle(img, (x,y), (x2,y2), self.cl_pr[class_id][1], 2)
            cv2.putText(img, self.cl_pr[class_id][0], (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, self.cl_pr[class_id][1], 2)

            if class_id == 2:
                ball_position = bbox

        return img, ball_position

class YOLOv8_Segmentation:
    def __init__(self, model_path, alpha=0.5, conf=0.35):
        self.model = YOLO(model_path)
        self.alpha = alpha
        self.conf = conf

    def detect(self, img, d, ball_position):
        height, width, channels = img.shape

        results = self.model.predict(conf=self.conf, source=img, save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        bboxes, class_ids, scores = [], [], []

        if result:
            for seg in result.masks.xyn:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            class_ids = np.array(result.boxes.cls.cpu(), dtype='int')
            scores = np.array(result.boxes.conf.cpu(), dtype='float').round(2)

        if len(d) == 0:
            for i in range(len(bboxes)):
                if class_ids[i] == 0: # collect only people
                    d[len(d)] = [bboxes[i], False, segmentation_contours_idx[i]]

        # return bboxes, class_ids, segmentation_contours_idx, scores

        overlay = img.copy()
        d = find_closest(d, bboxes, ball_position, segmentation_contours_idx)

        for i in range(len(d)):
            (x, y, x2, y2) = d[i][0]
            has_ball = d[i][1]
            seg = d[i][2]
            curr_color = (0,0,255)
            if has_ball:
                curr_color = (255,255,0)

            cv2.fillPoly(overlay, [seg], curr_color)
            # cv2.rectangle(img, (x,y), (x2,y2), curr_color, 2)
            # cv2.putText(img, str(i), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)



        # for bbox, class_id, seg, score in zip(bboxes, class_ids, segmentation_contours_idx, scores):
        #     if class_id == 0: # only people
        #         (x, y, x2, y2) = bbox
        #
        #         # cv2.fillPoly(overlay, [seg], (0,0,255))
        #         cv2.rectangle(img, (x,y), (x2,y2), (0,0,255), 2)

        return cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0), d
        # return img, d

# input_file = 'datasets/train_data/images/train/1.jpg'
# img = cv2.imread(input_file)
#
# model = YOLODetection('best.pt')
# print(model.detect(img))

# hyperparameters
input_file = '../videos/video_cut.mp4'
output_video_path = 'output.mp4'
det_weight = 'best.pt'
seg_weight = 'yolov8n-seg.pt'
alpha = 0.5 # transparency parameter

cap = cv2.VideoCapture(input_file)
model_det = YOLOv8_Detection(det_weight, class_parameters)
model_seg = YOLOv8_Segmentation(seg_weight)

frame_queue = []
vid_writer = []
vid_writer_forward = 0

seconds_forward = 3
seconds_back = 3

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
it = 0

d_tracking = {}
while True:
    print(it, '/', total_frames)
    it += 1

    ret, frame = cap.read()
    if not ret:
        break

    # if len(frame_queue) > fps*seconds_back:
    #     frame_queue.pop(0)
    # frame_queue.append(frame)

    # if model_det.has_score(frame):
    #     print('Score detected')
    frame, ball_position = model_det.detect(frame)
    frame, d_tracking = model_seg.detect(frame, d_tracking, ball_position)

    # cv2.imshow('image', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
