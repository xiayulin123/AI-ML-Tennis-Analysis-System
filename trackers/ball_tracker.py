from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        
        # convert the list to panda frame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        
        # interpolate the missing value
        df_ball_positions = df_ball_positions.interpolate()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        # to make the missing part between 2 detections be filled with value
        return ball_positions


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0] # tells model there are more than one frame keep persist

        ball_dict = {}


        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def  draw_boxes(self, video_frames, ball_detections):
        output_frame = []

        for frame, ball_dict in zip(video_frames, ball_detections): #zip help find 2 things in 2 paths same time
            # Draw the bounding box

            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Ball ID: {track_id}", (int(box[0]) - 20, int(box[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
                # 0 index minimum of x, 1 is minimum of y
                cv2.rectangle(frame, (int(x1) // 2, int(y1) // 2, int(x2) // 2, int(y2) // 2), (255, 0, 0), 2)

            output_frame.append(frame)
        return output_frame