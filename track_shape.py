import BallDetector
# import the necessary packages
import argparse
from collections import deque

import cv2
import imutils
import numpy as np

# parse CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=128, help="max buffer size (affects tracking line length)")
args = vars(ap.parse_args())

# initialise list of tracked points
pts = deque(maxlen=args["buffer"])

# load video
vs = cv2.VideoCapture(args["video"])

# loop through all video frames
while True:
    flag, frame = vs.read()
    if not flag:
        break

    # resize the frame, blur it, and convert it to the HSV color space
    #frame = imutils.resize(frame, width=1200)
    
    # convert the resized image to grayscale, blur it slightly, and threshold 
    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_OTSU)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #bd = BallDetector()

    # update the points queue 
    #pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", thresh)
    cv2.waitKey(1)

# Stop video processing
vs.release()
# close all windows
cv2.destroyAllWindows()



