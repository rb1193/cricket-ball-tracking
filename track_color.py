# import the necessary packages
import argparse
from collections import deque

import cv2

import tracking.contour
import tracking.frame
import tracking.video

# parse CLI arguments
args = tracking.cli.get_args()

# initialise list of tracked points
pts = deque(maxlen=args["buffer"])

# load video
vs = cv2.VideoCapture(args["video"])

# Perform ball tracking on the video
tracking.video.track_ball_by_color(vs, pts, args["width"])

# Stop video processing
vs.release()
# close all windows
cv2.destroyAllWindows()
