""" Video processing methods """
import cv2
import imutils
import tracking.frame

def track_ball_by_color(video, pts):
    """ Perform ball tracking on a video using color to identify the ball """

    # define colour bounds
    # provide ball color limits
    red_lower = (172, 66, 30)
    red_upper = (181, 200, 240)

    lastCenter = (0, 0)

    # loop through all video frames
    while True:
        flag, frame = video.read()
        if not flag:
            break

        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=1800)
        hsv = tracking.frame.preprocess(frame)
        mask = tracking.frame.mask_color(hsv, red_lower, red_upper)

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = tracking.frame.get_contours(mask)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then find the center
            c = max(cnts, key=cv2.contourArea)
            center = tracking.contour.get_center(c)

            trackedObject = (abs(center[0] - lastCenter[0]) < 10 and abs(center[1] - lastCenter[1]) < 10) or lastCenter == (0, 0)

            # Draw a circle around the contour
            tracking.frame.circle_contour(frame, c)

            # if the largest contour is not the object we're currently tracking, move to next frame
            #if not trackedObject:
            #    continue

            lastCenter = center

        # update the points queue 
        pts.appendleft(center)

        # draw the tracking line
        tracking.frame.draw_tracking_line(frame, pts)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
