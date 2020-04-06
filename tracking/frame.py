""" Frame processing utilities """
import imutils
import cv2
import numpy

def preprocess(frame):
    """ blur frame and convert it to the HSV color space """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

def mask_color(hsv, lower, upper):
    """ construct a mask for the color "red", then perform
    a series of dilations and erosions to remove any small
    blobs left in the mask """

    masked = cv2.inRange(hsv, lower, upper)
    masked = cv2.erode(masked, None, iterations=2)
    return cv2.dilate(masked, None, iterations=2)

def get_contours(mask):
    """ get the image contours exposed by a masked image"""
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)

def draw_tracking_line(frame, pts):
    """ loop over the set of tracked points and draw a line between them """
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(numpy.sqrt(len(pts) / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

def circle_contour(frame, contour):
    """ Draw a circle around the contour, if it's large enough """
    ((x, y), radius) = cv2.minEnclosingCircle(contour)
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)