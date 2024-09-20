from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2 

def main():
    # read videos
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detecting players
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path="models/last.pt")

    # Court line detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # choose the players
    player_detections = player_tracker.filtered_players(court_keypoints, player_detections)

    # Draw the output and box
    video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    video_frames = ball_tracker.draw_boxes(video_frames, ball_detections)
    video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)


    # Draw frame number on the top left
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    save_video(video_frames, "output_videos/output_video.avi")
if __name__ == "__main__":
    main()

