from collections import deque

import cv2
import tracking.cli
import tracking.video

# parse CLI arguments
args = tracking.cli.get_args()

# initialise list of tracked points
pts = deque(maxlen=args["buffer"])

# load video
vs = cv2.VideoCapture(args["video"])

tracking.video.track_ball_by_hough_circles(vs, args["width"])

# Stop video processing
vs.release()
# close all windows
cv2.destroyAllWindows()



