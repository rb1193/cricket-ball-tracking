""" CLI utilities """
import argparse

def get_args():
    """ Parse CLI args """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-b", "--buffer", type=int, default=128, help="max buffer size (affects tracking line length)")
    ap.add_argument("-w", "--width", type=int, default=1200, help="width to crop video frames to for processing")
    return vars(ap.parse_args())
