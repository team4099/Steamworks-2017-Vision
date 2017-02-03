#!/usr/bin/env python3
"""
    Given an IR image and a depth array, calculates angles to turn and distance to peg for FRC Steamworks 2017.
    Parth Oza
    Jagan Prem
    Oksana Tkach
    Some code originally from GRIP
    FRC Team 4099
"""
import cv2
import numpy
import freenect
from itertools import combinations
from frame_convert2 import *


DISTANCE_LIFT_TAPE = 10.25  #outside edges

class TooMuchInterferenceException(Exception):
    """
    The Exception raised when there is no goal in the image
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class GoalNotFoundException(Exception):
    """
    The Exception raised when there is no goal in the image
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

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
        real_depth = numpy.copy(depth_accumulator)
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
        return process_frame, real_depth

def combine_depth_frames(frame1, frame2):
    frame2[frame2 > 2046] = 0
    return numpy.bitwise_or(frame1, frame2)

def gaussian_blur(src):
    radius = 5
    ksize = int(6 * round(radius) + 1)
    return cv2.GaussianBlur(src, (ksize, ksize), round(radius))

def binary_threshold(src):
    return cv2.threshold(src, 20, 255, cv2.THRESH_BINARY)[1]

def hsv_threshold(src):
    hue, sat, val = [(0, 180), (0, 0), (128, 255)]
    out = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

def get_contours(src):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(src, mode=mode, method=method)
    print("len of contours:", len(contours))
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
            print("too many sides :/", len(contour))
            continue
        ratio = w / h
        if ratio < 0 or ratio > 0.7:
            print("ur ratio is trash", ratio)
            continue
        output.append(contour)
    return output

def y_filter_rectangles(rectangles, offset=100):
    median_y = numpy.median([rectangle[1] for rectangle in rectangles])
    return list(filter(lambda rectangle: abs(rectangle[1] - median_y) <= offset, rectangles))

def distance(p1, p2):
    print(p1, p2)
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))

def get_position_from_src(src):
    blur = gaussian_blur(src)

    threshold = binary_threshold(blur)
    cv2.imwrite("output/threshold.png", threshold)

    contours = get_contours(threshold)
    filtered_contours = filter_contours(contours)
    contour_drawn = cv2.cvtColor(numpy.copy(threshold), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_drawn, filtered_contours, -1, (0,255,0), 3)
    cv2.imwrite("output/contoured.png", contour_drawn)
 
    rectangles = []
    
    for contour in filtered_contours:
        rectangles.append(cv2.boundingRect(contour))

    rectangles = y_filter_rectangles(rectangles)

    if len(rectangles) < 2:
        raise GoalNotFoundException("yo no goals")
    elif len(rectangles) > 4:
        raise TooMuchInterferenceException("why is everything reflective :/")

    rectangles.sort()
    print("rectangles:", rectangles)

    if len(rectangles) == 3:
        combos = list(combinations((0, 1, 2), 2))
        pair = min((0, 1, 2), key=lambda i: distance(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        rectangles = [rectangles[combos[pair][0]], rectangles[combos[pair][1]]]
    elif len(rectangles) == 4:
        combos = list(combinations((0, 1, 2, 3), 2))
        pair1 = min(range(len(combos)), key=lambda i: distance(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        pair2 = len(combos) - pair1 - 1
        size = lambda i: rectangles[i][2] * rectangles[i][3]
        if size(combos[pair1][0]) + size(combos[pair1][1]) > size(combos[pair2][0]) + size(combos[pair2][1]):
            rectangles = [rectangles[combos[pair1][0]], rectangles[combos[pair1][1]]]
        else:
            rectangles = [rectangles[combos[pair2][0]], rectangles[combos[pair2][1]]]

    corner_set1, corner_set2 = rectangles[0], rectangles[1]
    print("corner set 1:", corner_set1)
    print("corner set 2:", corner_set2)
    final_coordinates = [[int(corner_set1[0]), int(corner_set1[1] + corner_set1[3] / 2)],
                         [int(corner_set2[0] + corner_set2[2]), int(corner_set2[1] + corner_set2[3] / 2)]]

    rectangles_drawn = cv2.cvtColor(numpy.copy(threshold), cv2.COLOR_GRAY2BGR)
    for rectangle in rectangles:
        cv2.rectangle(rectangles_drawn, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (255, 0, 0), 2)
    cv2.imwrite("output/rectangles.png", rectangles_drawn)
    # print(len(hull_position))
    return final_coordinates

def get_turning_angle(to_target_edge, left_distance, right_distance):
    """
    Params:
    to_target_edge = angle from front of robot to the nearest tape
    left_distance and right_distance = distance from robot to left/right tape
    """
    sides = (left_distance ** 2 + right_distance ** 2 - DISTANCE_LIFT_TAPE ** 2) / (2 * left_distance * right_distance)
    theta = numpy.arccos(sides)
    return to_target_edge + theta/2

def main():
    # while True:
    # image, depth = read_kinect_image(ir=True)
    # cv2.imwrite("output/ir.png", image)
    # cv2.imwrite("output/depth.png", pretty_depth_cv(depth))
    # numpy.save("depth.npy", depth)
    depth = numpy.load("depth.npy")
    image = cv2.cvtColor(cv2.imread("output/ir.png"), cv2.COLOR_BGR2GRAY)
    # rgb = read_kinect_image(ir=False)
    print(get_position_from_src(image))

if __name__ == '__main__':
    main()
