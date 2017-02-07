#!/usr/bin/env python3

import flask

from flask import Flask, render_template, Response
import vision_processing
import numpy
import freenect
from frame_convert2 import *
import cv2
import time

app = Flask(__name__)
process_flag = False
process_frame = None
depth_frame = None

last_frame_sent = 0


def get_frame():
    image = read_kinect_image()
    # print(image)
    # We are using Motion JPEG, but OpenCV defaults to capture raw images,
    # so we must encode it into JPEG in order to correctly display the
    # video stream.
    ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return jpeg.tobytes()


def combine_depth_frames(frame1, frame2):
    frame2[frame2 > 2046] = 0
    return numpy.bitwise_or(frame1, frame2)


def read_kinect_image(ir=False):
    if not ir:
        return video_cv(freenect.sync_get_video()[0])
    else:
        ir_feed = freenect.sync_get_video(0, format=freenect.VIDEO_IR_8BIT)
        ir_feed = ir_feed[1], ir_feed[0]
        # depth_feed = freenect.sync_get_depth()
        # ir_feed = freenect.sync_get_video(0, format=freenect.VIDEO_IR_8BIT)
        # cv2.imwrite("temp_video.png", ir_feed[1])
        depth_accumulator = freenect.sync_get_depth()[0]
        # print(real_depth)
        depth_accumulator[depth_accumulator > 2046] = 0
        for i in range(10):
            # print(depth_accumulator)
            depth_accumulator = combine_depth_frames(depth_accumulator, freenect.sync_get_depth()[0])
        real_depth = numpy.copy(depth_accumulator)
        depth_accumulator[depth_accumulator > 0] = 255
        # print(ir_feed)
        # depth_accumulator = depth_accumulator.astype()
        ir_feed = numpy.bitwise_and(depth_accumulator.astype(numpy.uint8), numpy.array(ir_feed[1])).astype(numpy.uint8)
        # cv2.imwrite("thing.png", frame_convert.pretty_depth_cv(ir_feed))
        process_frame = pretty_depth_cv(ir_feed)
        return process_frame, real_depth

def gen(write_flag=False):
    global last_frame_sent
    while True:
        frame = get_frame()
        if write_flag:
            cv2.imwrite("to_process.png", frame)
        if time.time() - last_frame_sent > 1/15:
            last_frame_sent = time.time()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_peg')
def get_peg():
    """
        Calculates the information the bot needs to move by in order
        to point the Kinect at the peg (turn_angle),  how many degrees off of horizontal
        the peg is, and how far forward it is

        :return: -1 if goal not found or the lateral and vertical offset in the
                following format:
                offset_angle,turn_angle,distance

    """
    frame, depth = read_kinect_image(ir=True)
    try:
        # print(frame)
        info = vision_processing.get_peg_info(frame, depth)
        # print(info)
        return ",".join([str(info["offset"]), str(info["turn"]), str(info["distance"])])
    except vision_processing.GoalNotFoundException:
        print("No goal found")
        return "-1", 503
    except vision_processing.TooMuchInterferenceException:
        print("No goal found (too much interference)")
        return "-1", 503
    except FileNotFoundError:
        print("No file found")
        return "-1", 503
    except Exception:
        print("something has gone horribly wrong")
        return "-1", 503


@app.route('/get_gear')
def get_gear():
    """
        Calculates the turning angle and distance needed for the bot to get to the gear

        :return: -1 if goal not found or the lateral and vertical offset in the
                following format:
                turn_angle,distance

    """
    frame = read_kinect_image()
    depth = read_kinect_image(ir=True)[1]
    try:
        info = vision_processing.get_gear_info(frame, depth)
        return ",".join([str(info["turn"]), str(info["distance"])])
    except vision_processing.GearNotFoundException:
        print("No gear found")
        return "-1", 503
    except FileNotFoundError:
        print("No file found")
        return "-1", 503
    except Exception as e:
        print("something has gone horribly wrong", e)
        return "-1", 503

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True, port=8080, threaded=True)
