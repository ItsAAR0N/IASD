# Test code for final Speed Estimation and Vehicle tracking using SUPERVISION
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 16/04/24

import argparse
import supervision as sv 
import numpy as np
from inference.models.utils import get_roboflow_model
from collections import defaultdict, deque
import cv2
import time

SOURCE = np.array(
    [
       [1252, 787], # A
       [2298, 803], # B
       [5039, 2159], # C
       [-550, 2159] # D
    ]
)

        # [708, 232], # A
        # [1173, 228], # B
        # [26, 525], # C
        # [1874, 542] # D

TARGET_WIDTH = 40 # RoI measures approx 25m x 250m long IRL
TARGET_HEIGHT = 121

TARGET = np.array(
    [
        [0, 0], # A'
        [TARGET_WIDTH - 1, 0], # B' 
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], # C'
        [0, TARGET_HEIGHT - 1], # D'
    ]
)

# Perspective Transformation using OpenCV "getPerspectiveTransform"
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        if transformed_points is None:
            return np.array([])  # Return an empty array if transformation fails
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation w/ Inference and SuperVision (from RoboFlow)"
    )
    parser.add_argument(
        "--source_video_path",
        default = "data/vehicles.mp4",
        required = False,
        help = "Path to the source video file",
        type = str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = get_roboflow_model("yolov8n-640") # YoLoV8n 640x input resolution.

    byte_track = sv.ByteTrack(frame_rate = video_info.fps) # Pass framerate because depends on info in constructor

    thickness = sv.calculate_dynamic_line_thickness(
        resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh = video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness = 4, color_lookup = sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale = text_scale, 
                                        text_thickness = thickness,
                                        text_position = sv.Position.BOTTOM_CENTER,
                                        color_lookup = sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(thickness = thickness, 
                                        trace_length = video_info.fps * 2, 
                                        position = sv.Position.BOTTOM_CENTER,
                                        color_lookup = sv.ColorLookup.TRACK)
    
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh = video_info.resolution_wh)
    view_transformer = ViewTransformer(source = SOURCE, target = TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps)) # Store coordinates of the car in the path using dict

    # Set window size and Mouse callback
    cv2.namedWindow("annotated_frame", cv2.WINDOW_NORMAL)  # Enable resizable window
    cv2.resizeWindow("annotated_frame", 1280, 720)  # Set initial window size
 
    for frame in frame_generator: # Pass frame by frame and run model
        start_calculation_time = time.time()  # Record start time for FPS calculation

        result = model.infer(frame)[0] # Run inference for every frame in for loop
        detections = sv.Detections.from_inference(result) # Convert result to Supervision object
        detections = detections[polygon_zone.trigger(detections)] # Whether detection is inside or outside zone
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Plug view transformer into existing detection pipeline
        points = view_transformer.transform_points(points = points).astype(int)

        # Store transformed coordinates
        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        for tracker_id in detections.tracker_id:
            # Wait to have enough data
            if len(coordinates[tracker_id]) < video_info.fps / 2: # < video_info.fps / 2
                labels.append(f"#{tracker_id}")
            else:
                # Calculate the speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                calc_time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / calc_time * 3.6
                if speed > 140:
                    labels.append(f"#{tracker_id} OVERSPEED {int(speed)} km/h")
                else:
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

        # labels = [ # Create labels then pass into label annotator
        #     f"x: {x}, y: {y}"
        #     for [x, y] in points
        # ]

        annotated_frame = frame.copy() # Call bounding box to annotator method
        annotated_frame = trace_annotator.annotate(
            scene = annotated_frame, detections = detections
        )

        # annotated_frame = sv.draw_polygon(annotated_frame, polygon = SOURCE, color = sv.Color.red())
        annotated_frame = bounding_box_annotator.annotate(
            scene = annotated_frame, detections = detections
        )
        annotated_frame = label_annotator.annotate(
            scene = annotated_frame, detections = detections, labels = labels
        )

        # annotated_frame_resized = cv2.resize(annotated_frame, (1280, 720))  # Resize the frame
        
        # Calculate FPS and add to frame
        current_time = time.time()
        fps = int(1 / (current_time - start_calculation_time))  # Calculate FPS
        cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Add FPS to the frame

        cv2.imshow("annotated_frame", annotated_frame)  # Show the resized frame

        if cv2.waitKey(1) == ord("q"): # Break
            break
    cv2.destroyAllWindows()
