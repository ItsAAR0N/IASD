# Code for obtaining coordinates of a road in preperation for perspective transformation
# Author: Aaron Shek, University of Hong Kong
# Date of last edit: 16/04/24

import argparse
import cv2

def parse_arguments() -> argparse.Namespace:
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Obtain pixel coordinates off video'
    )
    parser.add_argument('--video_path', 
                        default='data/videoplayback.mp4',
                        type=str, required=False, 
                        help = 'Path to video file'
                        )
    return parser.parse_args()

def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel Coordinates: ({x}, {y})")

if __name__ == "__main__":
    args = parse_arguments()

    cap = cv2.VideoCapture(args.video_path)

    # Set window size and mouse callback function
    cv2.namedWindow("blank_video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("blank_video", 1280, 720)
    cv2.setMouseCallback("blank_video", get_mouse_coordinates)

    while cap.isOpened():
        ret, frame = cap.read()
        # print("Reading frame...")  # Add this line for debugging
        if not ret: 
            break

        cv2.imshow("blank_video", frame)
        #print("Displaying frame...")  # Add this line for debugging

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()    
