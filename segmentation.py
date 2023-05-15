from ultralytics import YOLO
import numpy as np
import cv2

class YOLOSegmentation:
  def __init__(self, model_path):
    self.model = YOLO(model_path)

  def detect(self, img):
    height, width, channels = img.shape

    results = self.model.predict(source=img.copy(), save=False, save_txt=False)
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

    return bboxes, class_ids, segmentation_contours_idx, scores


input_file = '../videos/video1.mp4'
cap = cv2.VideoCapture(input_file)

alpha = 0.5  # Transparency factor.

# segmentation detector
model = YOLOSegmentation('yolov8m-seg.pt')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_video_path = 'output_video.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
it = 0

while True:
    print(it, '/', total_frames)
    it += 1

    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()

    bboxes, classes, segmentations, scores = model.detect(frame)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # print('bbox', bbox, 'class_id:', class_id, 'seg:', seg, 'score:', score)

        if class_id == 0: # only people
            (x, y, x2, y2) = bbox
            # cv2.rectangle(img, (x,y), (x2,y2), (0,0,255), 2)

            cv2.fillPoly(overlay, [seg], (0,0,255))

    # Following line overlays transparent rectangle
    # over the image
    image_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # out.write(image_new)

    cv2.imshow('image', image_new)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
