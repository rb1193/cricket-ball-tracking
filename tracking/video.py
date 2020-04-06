""" Video processing methods """
import cv2
import imutils
import numpy
import tracking.frame

def track_ball_by_color(video, pts, res=1200):
    """ Perform ball tracking on a video using color to identify the ball """

    # define colour bounds
    red_lower = (172, 100, 30)
    red_upper = (181, 200, 240)

    lastCenter = (0, 0)

    # loop through all video frames
    while True:
        flag, frame = video.read()
        if not flag:
            break

        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=res)
        hsv = tracking.frame.preprocess(frame)
        mask = tracking.frame.mask_color(hsv, red_lower, red_upper)

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = tracking.frame.get_contours(mask)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then find the center
            c = max(cnts, key=cv2.contourArea)
            if not tracking.contour.is_circular(c):
                continue
            
            center = tracking.contour.get_center(c)

            #trackedObject = (abs(center[0] - lastCenter[0]) < 10 and abs(center[1] - lastCenter[1]) < 10) or lastCenter == (0, 0)

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

def track_ball_by_hough_circles(video, res=1200):
    while True:
        flag, frame = video.read()
        if not flag:
            break

        # Convert to greyscale and blur
        frame = imutils.resize(frame, width=res)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=100, param2=22,
                                minRadius=7, maxRadius=13)
        
        if circles is not None:
            print(len(circles))
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = numpy.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                print (x, y, r)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(gray, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        cv2.imshow("detected circles", gray)
        cv2.waitKey(1)
