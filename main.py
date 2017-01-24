#!/usr/bin/env python3
"""
Code originally from GRIP
"""
import cv2
import numpy
import freenect
from frame_convert2 import *

def gaussian_blur(src):
    radius = 3
    ksize = int(6 * round(radius) + 1)
    return cv2.GaussianBlur(src, (ksize, ksize), round(radius))

def hsv_threshold(src):
    hue, sat, val = [(0, 180), (0, 0), (128, 255)]
    out = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

def get_contours(src):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(src, mode=mode, method=method)
    print(len(contours))
    return contours

def filter_contours(input_contours):
    output = []
    for contour in input_contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w < 0 or w > 1000:
            continue
        if h < 0 or h > 1000:
            continue
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        if cv2.arcLength(contour, True) < 0:
            continue
        hull = cv2.convexHull(contour)
        solid = 100 * area / cv2.contourArea(hull)
        if solid < 0 or solid > 100:
            continue
        if len(contour) < 0 or len(contour) > 200:
            continue
        ratio = w / h
        if ratio < 0 or ratio > 0.5:
            continue
        output.append(contour)
    return output

def find_hulls(input_contours):
    return [cv2.convexHull(contour) for contour in input_contours]

def get_position_from_src(src):
    blur = gaussian_blur(src)
    contours = get_contours(blur)
    filtered_contours = filter_contours(contours)
    hull_position = find_hulls(filtered_contours)
    print(len(hull_position))
    return contours

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
        depth_accumulator[depth_accumulator > 2046] = 0
        for i in range(10):
            # print(depth_accumulator)
            depth_accumulator = combine_depth_frames(depth_accumulator, freenect.sync_get_depth()[0])
        depth_accumulator[depth_accumulator > 0] = 255
        # print(ir_feed)
        # depth_accumulator = depth_accumulator.astype()
        ir_feed = numpy.bitwise_and(depth_accumulator.astype(numpy.uint8), numpy.array(ir_feed[1])).astype(numpy.uint8)
        # cv2.imwrite("thing.png", frame_convert.pretty_depth_cv(ir_feed))
        process_frame = pretty_depth_cv(ir_feed)
        return process_frame

def main():
    # while True:
    image = read_kinect_image(ir=True)
    # rgb = read_kinect_image(ir=False)
    # image = cv2.imread("potato.png")
    cv2.imwrite("morepotato.png", image)
    print(image)
    contoured = numpy.copy(image)
    cv2.drawContours(contoured, get_position_from_src(image), -1, (0, 254, 0))
    cv2.imwrite("potato.png", contoured)
    # cv2.imshow('potato chip', get_position_from_src(image))
    # cv2.waitKey(10000)
    pass

if __name__ == '__main__':
    main()
