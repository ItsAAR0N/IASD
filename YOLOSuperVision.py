# Test code for final Speed Estimation and Vehicle tracking 
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 05/04/24

import argparse
import supervision as sv 
import numpy as np
from inference.models.utils import get_roboflow_model
from collections import defaultdict, deque
import cv2

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation w/ Inference and SuperVision (from RoboFlow)"
    )
    parser.add_argument(
        "--source_video_path",
        required = True,
        help = "Path to the source video file",
        type = str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = get_roboflow_model("yolov8x-640") # YoLoV8 640x input resolution.

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

    for frame in frame_generator:
        result = model.infer(frame)[0] # Run inference for every frame in for loop
        detections = sv.Detections.from_inference(result) # Convert result to Supervision object
        detections = detections[polygon_zone.trigger(detections)] # Whether detection is inside or outside zone
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points = points).astype(int)

        labels = []
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
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
        cv2.imshow("annotated_frame", annotated_frame) # Use CV2 to visualize
        if cv2.waitKey(1) == ord("q"): # Break
            break
    cv2.destroyAllWindows()