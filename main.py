from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker


def main():
    # read videos
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detecting players
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path="models/last.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections.pkl")

    # Draw the output and box
    video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    video_frames = ball_tracker.draw_boxes(video_frames, ball_detections)


  

    save_video(video_frames, "output_videos/output_video.avi")
if __name__ == "__main__":
    main()

