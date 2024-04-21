# Test code for final Speed Estimation and Vehicle tracking using ULTRALYTICS
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 21/04/24
# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors
import cv2
import argparse
from collections import defaultdict
from shapely.geometry import LineString, Point, Polygon
from time import time
import numpy as np

check_requirements("shapely>=2.0.0")

class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2 or 4")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Draw bounding box
            # self.annotator.box_label(box, label=f"{track_id}:{self.names[cls]}", color=colors(int(cls), True))

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(
                    track_line, color=self.track_color, track_thickness=self.track_thickness
                )

            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            # Count objects
            if len(self.reg_pts) == 4:
                if (
                    prev_position is not None
                    and self.counting_region.contains(Point(track_line[-1]))
                    and track_id not in self.counting_list
                ):
                    self.counting_list.append(track_id)
                    if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        self.in_counts += 1
                    else:
                        self.out_counts += 1

            elif len(self.reg_pts) == 2:
                if prev_position is not None:
                    distance = Point(track_line[-1]).distance(self.counting_region)
                    if distance < self.line_dist_thresh and track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                        else:
                            self.out_counts += 1

        incount_label = f"Cars Out : {self.in_counts}"
        outcount_label = f"Cars in : {self.out_counts}"

        # Display counts based on user choice
        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = outcount_label
        elif not self.view_out_counts:
            counts_label = incount_label
        else:
            counts_label = f"{incount_label} {outcount_label}"

        if counts_label is not None:
            self.annotator.count_labels(
                counts=counts_label,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
            )

    def display_frames(self, window_name="Ultralytics Object Counter and Speed Estimation"):
        """Display frame."""
        if self.env_check:
            # cv2.namedWindow("Ultralytics YOLOv8 Object Counter")
            # cv2.imshow(window_name, self.im0)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(
                    "Ultralytics YOLOv8 Object Counter", self.mouse_event_for_region, {"region_points": self.reg_pts}
                )
            cv2.imshow(window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image

        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return im0
        self.extract_and_process_tracks(tracks)

        if self.view_img:
            self.display_frames()
        return self.im0

class SpeedEstimator:
    """A class to estimation speed of objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the speed-estimator class with default values for Visual, Image, track and speed parameters."""
        print("Speed Estimator Initiated.")
        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False

        # Region information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.region_thickness = 3

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.line_thickness = 2
        self.trk_history = defaultdict(list)

        # Speed estimator information
        self.current_time = 0
        self.dist_data = {}
        self.trk_idslist = []
        self.spdl_dist_thresh = 10
        self.trk_previous_times = {}
        self.trk_previous_points = {}

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        reg_pts,
        names,
        view_img=False,
        line_thickness=2,
        region_thickness=5,
        spdl_dist_thresh=10,
    ):
        """
        Configures the speed estimation and display parameters.

        Args:
            reg_pts (list): Initial list of points defining the speed calculation region.
            names (dict): object detection classes names
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            region_thickness (int): Speed estimation region thickness
            spdl_dist_thresh (int): Euclidean distance threshold for speed line
        """
        if reg_pts is None:
            print("Region points not provided, using default values")
        else:
            self.reg_pts = reg_pts
        self.names = names
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.region_thickness = region_thickness
        self.spdl_dist_thresh = spdl_dist_thresh

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Store track data.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
        """
        track = self.trk_history[track_id]
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        track.append(bbox_center)

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Plot track and bounding box.

        Args:
            track_id (int): object track id.
            box (list): object bounding box data
            cls (str): object class name
            track (list): tracking history for tracks path drawing
        """
        speed_label = f"{int(self.dist_data[track_id])}kmh" if track_id in self.dist_data else self.names[int(cls)]

        # Determine speed condition
        if track_id in self.dist_data:
            if self.dist_data[track_id] > args.speed_limit:
                bbox_color = (0, 0, 255)  # Beyond certain speed (red)
                speed_label = f"{int(self.dist_data[track_id])}kmh OVERSPEED!" if track_id in self.dist_data else self.names[int(cls)]

            else:
                bbox_color = (0, 255, 0)  # Under certain speed (green)
        else:
            bbox_color = (128, 128, 128)  # Default color if speed data is not available

        # bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        self.annotator.box_label(box, speed_label, bbox_color, (0, 0, 0)) # Black font

        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculation of object speed.

        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """

        if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
            return
        if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
            direction = "known"

        elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
            direction = "known"

        else:
            direction = "unknown"

        if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference
                self.dist_data[trk_id] = speed

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Calculate object based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple): Color to use when drawing regions.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return im0
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=region_color, thickness=self.region_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self, window_name="Ultralytics Object Counter and Speed Estimation"):
        """Display frame."""
        cv2.imshow(window_name, self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Speed Estimation using ULTRALYTICS'
    )
    parser.add_argument('--video_path', default='./data/two.mp4', 
                        type=str, required=False, help = 'Path to video file'),
    parser.add_argument('--height_fraction', default= 0.7, 
                        type=float, required=False, help = 'Fraction of height of video frame (Y-axis)')                        
    parser.add_argument('--video_no', default= 'X', 
                        type=str, required=False, help = 'Video No.')   
    parser.add_argument('--speed_limit', default= 100, 
                        type=int, required=False, help = 'Speed Limit') 
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    model = YOLO("yolov8n.pt") # Use nano model (~3.2 million parameters)
    names = model.model.names

    cap = cv2.VideoCapture(args.video_path) # Can alternative use web cam too (opens the video)

    assert cap.isOpened(), "Error reading video file"

    # Get the width, height, and frames per second (fps) from the video capture object
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Video Writer
    video_writer = cv2.VideoWriter("./output/Speed_estimation_{0}.mp4".format(args.video_no),
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w,h))
    
    fraction_of_height = args.height_fraction
    line_pts = [(0, int(fraction_of_height * h)), (w, int(fraction_of_height * h))] # Line location

    # Init speed-estimation obj
    speed_obj = SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts, # Points defining the region area
                       names=names, # Classes names
                       view_img=True) # Display frame with counts

    # Init Object Counter
    counter = ObjectCounter()
    counter.set_args(view_img=True,
                    reg_pts=line_pts,
                    classes_names=model.names,
                    draw_tracks=True) # line_thickness=2

    while cap.isOpened():

        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been done.")
            break

        # Calculate FPS
        start_time = time()

        tracks = model.track(im0, persist = True, show = False,tracker="bytetrack.yaml") # Source directory for vid; Persisting tracks between frames 

        im0 = speed_obj.estimate_speed(im0, tracks)
        im0 = counter.start_counting(im0, tracks)

        # Calculate FPS
        end_time = time()
        fps = 1.0 / (end_time - start_time)
        print("FPS: {0}".format(str(round(fps, 2))))

        # Display FPS in the video window
        cv2.putText(im0, f"FPS: {str(round(fps,2))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Add FPS to the frame

        speed_obj.display_frames()

        video_writer.write(im0)

        # Allow exit method
        if cv2.waitKey(1) == ord("q"): # Break
            break
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()