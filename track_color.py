# import the necessary packages
import argparse
from collections import deque

import cv2

import tracking.contour
import tracking.frame
import tracking.video

# parse CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=128, help="max buffer size (affects tracking line length)")
args = vars(ap.parse_args())

# initialise list of tracked points
pts = deque(maxlen=args["buffer"])

# load video
vs = cv2.VideoCapture(args["video"])

# Perform ball tracking on the video
tracking.video.track_ball_by_color(vs, pts)

# Stop video processing
vs.release()
# close all windows
cv2.destroyAllWindows()
