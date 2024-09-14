from ultralytics import YOLO
import cv2
import pickle

class PointTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        point_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                point_detections = pickle.load(f)
            return point_detections
        
        for frame in frames:
            point_dict = self.detect_frame(frame)
            point_detections.append(point_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(point_detections, f)

        return point_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # tells model there are more than one frame keep persist
        id_name_dict = results.names

        point_dict = {}


        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                point_dict[track_id] = result
        
        return point_dict

    def  draw_boxes(self, video_frames, point_detections):
        output_frame = []

        for frame, point_dict in zip(video_frames, point_detections): #zip help find 2 things in 2 paths same time
            # Draw the bounding box

            for track_id, box in point_dict.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Point ID: {track_id}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
                # 0 index minimum of x, 1 is minimum of y
                cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)

            output_frame.append(frame)
        return output_frame