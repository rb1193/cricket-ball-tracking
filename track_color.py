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

# provide ball color limits
redLower = (172, 66, 30)
redUpper = (181, 200, 240)

# initialise list of tracked points
pts = deque(maxlen=args["buffer"])
lastCenter = (0, 0)

# load video
vs = cv2.VideoCapture(args["video"])

# loop through all video frames
while True:
    flag, frame = vs.read()
    if not flag:
        break

    # resize the frame, blur it, and convert it to the HSV color space
    #frame = imutils.resize(frame, width=1800)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        trackedObject = (abs(center[0] - lastCenter[0]) < 10 and abs(center[1] - lastCenter[1]) < 10) or lastCenter == (0, 0)

        # only proceed if the radius meets a minimum size and it's part of the same object
        if radius > 3:
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # if the largest contour is not the object we're currently tracking, move to next frame
        #if not trackedObject:
        #    continue

        lastCenter = center

    # update the points queue 
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(hsv, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

# Stop video processing
vs.release()
# close all windows
cv2.destroyAllWindows()
