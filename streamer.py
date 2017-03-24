#!/usr/bin/env python3
"""
Gets images from Kinect and streams them using Flask. Also, uses vision processing to find areas of interest for
FRC Steamworks 2017.

Parth Oza
Kent Ma
Jagan Prem
Original Motion JPEG Streamer code (c) Miguel Grinberg (https://github.com/miguelgrinberg/flask-video-streaming)
FRC Team 4099
"""

from flask import Flask, render_template, Response
from frame_convert2 import *
from functools import wraps
import vision_processing
import numpy
import freenect
import cv2
import time
import traceback
import os
import sys
import subprocess
import signal
import errno

SPF = 1 / 30
FRAMES_PER_VIDEO_FILE = SPF * 15

app = Flask(__name__)
rgb_frame = None
depth_frame = None
ir_frame = None

last_frame_sent = 0

frames_stored = float("inf")

rgb_writer = cv2.VideoWriter()
depth_writer = cv2.VideoWriter()
ir_writer = cv2.VideoWriter()

get_ir = False

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

def encode_frame(image, quality=30):
    """
    Encodes OpenCV image into JPEG format with set compression
    :param image: image to encode (numpy array)
    :param quality: Percentage of quality to preserve (inverse of compression amount)
    :return: JPEG bytes of image given
    """
    image = numpy.copy(image)
    lined_image1 = cv2.line(image, (258, 0), (86, 480), (255, 0, 119), thickness=4, lineType=cv2.LINE_AA)
    lined_image2 = cv2.line(lined_image1, (308, 0), (522, 480), (255, 0, 119), thickness=4, lineType=cv2.LINE_AA)
    ret, jpeg = cv2.imencode(".jpg", lined_image2, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def combine_depth_frames(frame1, frame2):
    """
    Given two depth arrays, combines them to eliminate as many zero values as possible
    :param frame1: original frame to add to
    :param frame2: new frame to combine into frame1
    :return: the combined image
    """
    frame2[frame2 > 2046] = 0
    return numpy.bitwise_or(frame1, frame2)


@timeout(5)
def read_kinect_images(ir=True):
    """
    Gets rgb, ir, and combined depth images from Kinect
    :return: rgb, ir, depth images in that order as a tuple
    """
    if not ir:
        return video_cv(freenect.sync_get_video()[0])
    ir_feed = freenect.sync_get_video(0, format=freenect.VIDEO_IR_8BIT)
    ir_feed = ir_feed[1], ir_feed[0]
    depth_accumulator = freenect.sync_get_depth()[0]
    depth_accumulator[depth_accumulator > 2046] = 0
    for i in range(10):
        depth_accumulator = combine_depth_frames(depth_accumulator, freenect.sync_get_depth()[0])
    real_depth = numpy.copy(depth_accumulator)
    depth_accumulator[depth_accumulator > 0] = 255
    # print(ir_feed)
    # depth_accumulator = depth_accumulator.astype()
    ir_feed = numpy.bitwise_and(depth_accumulator.astype(numpy.uint8), numpy.array(ir_feed[1])).astype(numpy.uint8)
    # cv2.imwrite("thing.png", frame_convert.pretty_depth_cv(ir_feed))
    ir_frame = pretty_depth_cv(ir_feed)
    return ir_frame, real_depth


def gen():
    """
    Generator for motion JPEG format - keeps returning next frame in stream - caps at 35 fps
    :return: Next frame of stream encoded as frame of Motion JPEG stream
    """
    global last_frame_sent, rgb_frame, frames_stored, get_ir, ir_frame, depth_frame
    while True:
        # frames_stored += 1
        try:
            if get_ir:
                ir_frame, depth_frame = read_kinect_images(ir=True)
                get_ir = False
            if time.time() - last_frame_sent > SPF:
                rgb_frame = read_kinect_images(ir=False)
                frame = encode_frame(rgb_frame)
                last_frame_sent = time.time()
                yield (b"--frame\nContent-Type: image/jpeg\n\n" + frame + b"\n\r\n")
        except:
            print("it died pls")
            restart()
            pass


@app.route("/video_feed")
def video_feed():
    """
    Flask endpoint for video stream
    :return: Properly formed HTTP response for next frame in multipart
    """
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_lift")
def get_lift():
    """
    Calculates the information the bot needs to move by in order
    to point the Kinect at the lift (turn_angle),  how many degrees off of horizontal
    the lift is, and how far forward it is

    :return: -1 if goal not found or the lateral and vertical offset in the
            following format:
            offset_angle,turn_angle,distance

    """
    global ir_frame, depth_frame, get_ir
    print("gotten the lifterino?")
    try:
        # print(frame)
        get_ir = True
        while get_ir:
            time.sleep(0.1)
        print("Survived condition check")
        # ir_frame, depth_frame = read_kinect_images()
        # cv2.imwrite("potato.png", ir_frame)
        # print(ir_frame)

        info = vision_processing.get_lift_info(ir_frame, depth_frame)
        # print(info)
        return ",".join([str(info["offset"]), str(info["turn"]), str(info["distance"])])
    except vision_processing.LiftNotFoundException:
        print("No lift found, trying just depth")
        try:
            info = vision_processing.get_lift_info_just_depth(depth_frame)
            return ",".join([str(info["offset"]), str(info["turn"]), str(info["distance"])])
        except Exception as e:
            print("Tried to do it as depth and didn't work", e)
            return "-1", 503
    except vision_processing.TooMuchInterferenceException:
        print("No lift found (too much interference)")
        return "-1", 503
    except Exception as e:
        print("something has gone horribly wrong", e)
        traceback.print_exc()
        return "-1", 503


@app.route("/get_gear")
def get_gear():
    """
    Calculates the turning angle and distance needed for the bot to get to the gear

    :return: -1 if goal not found or the lateral and vertical offset in the
            following format:
            turn_angle,distance

    """
    global rgb_frame, depth_frame
    try:
        info = vision_processing.get_gear_info(rgb_frame, depth_frame)
        return ",".join([str(info["turn"]), str(info["distance"])])
    except vision_processing.GearNotFoundException:
        print("No gear found")
        return "-1", 503
    except Exception as e:
        print("something has gone horribly wrong", e)
        traceback.print_exc()
        return "-1", 503

@app.route("/restart")
def restart():
    return subprocess.check_output(['/sbin/initctl', 'restart', 'vision']), 500

@app.errorhandler(Exception)
def all_exception_handler(error):
    """
    Flask endpoint for catching all errors to prevent crashes on robot or streamer
    :param error: Error that was caught
    :return: -1 and error code
    """
    print(error)
    restart()
    return "-1", 500


@app.route("/")
def index():
    """
    The Flask endpoint for main page for video streamer
    :return: the rendered form of the index.html page
    """
    return render_template("index.html")


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True, port=8080, threaded=True)
