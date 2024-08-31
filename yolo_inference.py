from ultralytics import YOLO

# model = YOLO('models/trained_model.pt')
model = YOLO('yolov8x')


# result = model.predict("input_videos/test_image.jpeg", conf=0.2, save = True)
# result = model.predict("input_videos/input_video.mp4", conf=0.2, save = True)

result = model.track("input_videos/test_image.jpeg", conf=0.2, save = True)

# print(result)
# print("Boxes:\n")
# for box in result[0].boxes:
#     print(box)

