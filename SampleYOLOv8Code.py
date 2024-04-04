# Test code for YOLOv8 algorithm provided by ultralytics
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 08/03/24

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

model = YOLO('yolov8x.pt')

# results = model.track(source = 0, show = True, tracker = "bytetrack.yaml")

# results = model.predict(source=0, verbose=False, stream=True)

results = model(source=0, stream=True, show = True)  # generator of Results objects
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs