# Test code for object counting using ULTRALYTICS
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 21/04/24

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import argparse

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Obtain pixel coordinates off video'
    )
    parser.add_argument('--video_path', 
                        default='data/Two.mp4',
                        type=str, required=False, 
                        help = 'Path to video file'
                        )
    parser.add_argument('--height_fraction', default= 0.7, 
                        type=float, 
                        required=False, 
                        help = 'Fraction of height of video frame (Y-axis)'
                        )  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model = YOLO("yolov8n.pt") # Utilize Yolo V8 nano model (~3.2 million parameters)
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define line points
    fraction_of_height = args.height_fraction
    line_points = [(0, int(fraction_of_height * h)), (w, int(fraction_of_height * h))] # Line location

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi",
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (w, h))

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                    reg_pts=line_points,
                    classes_names=model.names,
                    draw_tracks=True,
                    line_thickness=2)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()