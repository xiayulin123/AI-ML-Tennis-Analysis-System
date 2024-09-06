from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        player_detections = []
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True) # tells model there are more than one frame keep persist
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
