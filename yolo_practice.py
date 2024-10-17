from ultralytics import YOLO

#model = YOLO("yolov8x") to load video first vieo trained
model = YOLO('training/models/best.pt')

#results = model.predict('input_videos/08fd33_4.mp4', save=True) to load and save video results
results = model.predict('input_videos/08fd33_4.mp4', save = True)
print(results[0])
print('===========')
for box in results[0].boxes:
    print(box)