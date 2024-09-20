from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def filtered_players(self, keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_player(keypoints, player_detections_first_frame)
        filtered_player = []
        for player_dictance in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dictance.items() if track_id in chosen_player}
            filtered_player.append(filtered_player_dict)
        return filtered_player

    def choose_player(self, keypoints, player_detections_first_frame):
        distances = []
        if isinstance(player_detections_first_frame, list):
            combined_player_detections = {}
            for detection in player_detections_first_frame:
                combined_player_detections.update(detection)
        else:
            combined_player_detections = player_detections_first_frame
        # Ensure that player_detections_first_frame is a dictionary
        # Iterate over the dictionary's key-value pairs (track_id and bbox)
        for track_id, bbox in combined_player_detections.items():
            # Get the player center
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')

            # Calculate the minimum distance to keypoints
            for i in range(0, len(keypoints), 2):
                keypoint = (keypoints[i], keypoints[i + 1])
                # Calculate the distance between player_center and keypoint
                distance = measure_distance(player_center, keypoint)
                if distance < min_distance:
                    min_distance = distance

            # Append (track_id, min_distance) as a tuple to the distances list
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order by the second element (min_distance)
        distances.sort(key=lambda x: x[1])

        # Choose the first 2 trackers with the smallest distance
        chosen_players = [distances[0][0], distances[1][0]] if len(distances) >= 2 else []

        return chosen_players



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # tells model there are more than one frame keep persist
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

    def  draw_boxes(self, video_frames, player_detections):
        output_frame = []
        print(f"here is the thing: {player_detections}")

        for frame, player_dict in zip(video_frames, player_detections): #zip help find 2 things in 2 paths same time
            # Draw the bounding box
            for track_id, box in player_dict.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Player ID: {track_id}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2)
                # 0 index minimum of x, 1 is minimum of y
                cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)

            output_frame.append(frame)
        return output_frame