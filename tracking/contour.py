""" Contour utilities """
import cv2

def get_center(contour):
    """ Get the center of a contour """
    M = cv2.moments(contour)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
