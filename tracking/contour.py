""" Contour utilities """
import cv2

def get_center(contour):
    """ Get the center of a contour """
    M = cv2.moments(contour)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def is_circular(contour):
    """ Determine if a contour is roughly circular """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)
    return len(approx) > 8
