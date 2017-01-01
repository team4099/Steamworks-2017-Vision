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
import math
import freenect
from itertools import combinations
from frame_convert2 import *
import copy
import time
import glob
import timeit


DISTANCE_LIFT_TAPE_METERS = 0.26  #outside edges
FOV_OF_CAMERA = math.radians(57)
IMAGE_WIDTH_PX = 640


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

class GearNotFoundException(Exception):
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
        print(real_depth)
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


def gaussian_blur(src, radius):
    ksize = int(6 * round(radius) + 1)
    return cv2.GaussianBlur(src, (ksize, ksize), round(radius))


def peg_binary_threshold(src):
    return cv2.threshold(src, 15, 255, cv2.THRESH_BINARY)[1]


def gear_hsv_threshold(src):
    hue, sat, val = [(11, 180), (170, 255), (124, 255)]
    out = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))


def get_contours(src):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(src, mode=mode, method=method)
    # print("len of contours:", len(contours))
    return contours


def peg_filter_contours(input_contours):
    output = []
    for contour in input_contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        if len(contour) < 0 or len(contour) > 200:
            # print("too many sides :/", len(contour))
            continue
        ratio = w / h
        if ratio < 0 or ratio > 0.7:
            # print("ur ratio is trash", ratio)
            continue
        output.append(contour)
    return output


def y_filter_rectangles(rectangles, offset=100):
    median_y = numpy.median([rectangle[1] for rectangle in rectangles])
    return list(filter(lambda rectangle: abs(rectangle[1] - median_y) <= offset, rectangles))


def distance_squared(p1, p2):
    print("p1, p2:", p1, p2)
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))


def get_peg_position_from_src(src):
    blur = gaussian_blur(src, 5)

    threshold = peg_binary_threshold(blur)
    # cv2.imwrite("output/threshold.png", threshold)

    contours = get_contours(threshold)
    filtered_contours = peg_filter_contours(contours)
    # contour_drawn = cv2.cvtColor(numpy.copy(threshold), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(contour_drawn, filtered_contours, -1, (0,255,0), 3)
    # cv2.imwrite("output/contoured.png", contour_drawn)
 
    rectangles = []
    
    for contour in filtered_contours:
        rectangles.append(cv2.boundingRect(contour))

    rectangles = y_filter_rectangles(rectangles)

    if len(rectangles) < 2:
        raise GoalNotFoundException("yo no goals")
    elif len(rectangles) > 4:
        raise TooMuchInterferenceException("why is everything reflective :/")

    rectangles.sort()
    # print("rectangles:", rectangles)

    if len(rectangles) == 3:
        combos = list(combinations((0, 1, 2), 2))
        pair = min((0, 1, 2), key=lambda i: distance_squared(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        rectangles = [rectangles[combos[pair][0]], rectangles[combos[pair][1]]]
    elif len(rectangles) == 4:
        combos = list(combinations((0, 1, 2, 3), 2))
        pair1 = min(range(len(combos)), key=lambda i: distance_squared(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        pair2 = len(combos) - pair1 - 1
        size = lambda i: rectangles[i][2] * rectangles[i][3]
        if size(combos[pair1][0]) + size(combos[pair1][1]) > size(combos[pair2][0]) + size(combos[pair2][1]):
            rectangles = [rectangles[combos[pair1][0]], rectangles[combos[pair1][1]]]
        else:
            rectangles = [rectangles[combos[pair2][0]], rectangles[combos[pair2][1]]]

    corner_set1, corner_set2 = rectangles[0], rectangles[1]
    # print("corner set 1:", corner_set1)
    # print("corner set 2:", corner_set2)
    final_coordinates = [[int(corner_set1[0]), int(corner_set1[1] + corner_set1[3] / 2)],
                         [int(corner_set2[0] + corner_set2[2]), int(corner_set2[1] + corner_set2[3] / 2)]]

    rectangles_drawn = cv2.cvtColor(numpy.copy(threshold), cv2.COLOR_GRAY2BGR)
    for rectangle in rectangles:
        cv2.rectangle(rectangles_drawn, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (255, 0, 0), 2)
    # cv2.imwrite("output/rectangles.png", rectangles_drawn)
    # print(len(hull_position))
    return final_coordinates


def get_turning_angle(center_x):
    return angle_at_x(center_x) - FOV_OF_CAMERA / 2


def get_offset_angle(distance_1, distance_2, angle_1, angle_2):
    f = (FOV_OF_CAMERA / 2)
    # if distance_1 > distance_2:
    #     distance_1, distance_2 = distance_2, distance_1
    #     angle_1, angle_2 = angle_2, angle_1
    return numpy.arcsin((distance_2 - (distance_1 * numpy.cos(f - angle_1) / numpy.cos(f - angle_2))) * numpy.cos(f - angle_2) / DISTANCE_LIFT_TAPE_METERS)


def kinect_depth_to_meters(kinect_depth):
    return 1 / (3.14674327 + -0.0028990952 * kinect_depth)


def depth_at_pixel(depth_array, pixel, direction=1, side_direction=-1):
    new_pixel = numpy.copy(pixel)
    while 0 < new_pixel[1] < depth_array.shape[0] - 1 and depth_array[int(new_pixel[1]), int(new_pixel[0])] == 0:
        # print("stuck in first loop here :/")
        new_pixel[1] += direction
    if depth_array[int(new_pixel[1])][int(new_pixel[0])] == 0:
        new_pixel = numpy.copy(pixel)
        while 0 < new_pixel[0] < depth_array.shape[1] - 1 and depth_array[new_pixel[1], new_pixel[0]] == 0:
            new_pixel[0] += side_direction
            # print("stuck in first loop here :)")    
    # print("raw depth:", depth_array[new_pixel[1], new_pixel[0]])
    return kinect_depth_to_meters(depth_array[int(new_pixel[1]), int(new_pixel[0])])


def angle_at_x(x_value):
    return (x_value / IMAGE_WIDTH_PX) * FOV_OF_CAMERA


def get_peg_info(image, depth):
    # print("potato")
    # cv2.imwrite("output/ir.png", image)
    # cv2.imwrite("output/depth.png", pretty_depth_cv(numpy.copy(depth)))
    position = get_peg_position_from_src(image)
    depth_at_1 = depth_at_pixel(depth, position[0], side_direction=1)
    depth_at_2 = depth_at_pixel(depth, position[1])
    angle_at_1 = angle_at_x(position[0][0])
    angle_at_2 = angle_at_x(position[1][0])
    offset_angle = math.degrees(get_offset_angle(depth_at_1, depth_at_2, angle_at_1, angle_at_2))
    turning_angle = math.degrees(get_turning_angle((position[0][0] + position[1][0]) / 2))
    return {"offset": offset_angle, "turn": turning_angle, "distance": (depth_at_1 + depth_at_2) / 2}


def is_in_ellipse(ellipse, point):
    x, y = point
    xc = ellipse[0][0]
    yc = ellipse[0][1]
    a = ellipse[1][0] / 2
    b = ellipse[1][1] / 2
    theta = math.radians(ellipse[2])
    sint = numpy.sin(theta)
    cost = numpy.cos(theta)
    A = a ** 2 * sint ** 2 + b ** 2 * cost ** 2
    B = 2 * (b ** 2 - a ** 2) * sint * cost
    C = a ** 2 * cost ** 2 + b ** 2 * sint ** 2
    D = -2 * A * xc - B * yc
    E = -B * xc - 2 * C * yc
    F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - a ** 2 * b ** 2
    return A * x ** 2 + B * x * y + C * y ** 2 + D * x + E * y + F <= 0


def get_gear_info(image, depth):
    blur = gaussian_blur(image, 5)
    threshold = gear_hsv_threshold(blur)
    contours = get_contours(threshold)
    hulls = [cv2.convexHull(contour) for contour in contours if cv2.contourArea(contour) >= 100]

    if len(hulls) == 0:
        raise GearNotFoundException("No hulls found :(")

    ellipses = [cv2.fitEllipse(hull) for hull in hulls]
    best_index = 0
    fit_number = float("inf")

    for i, hull in enumerate(hulls):
        ellipse = ellipses[i]
        center = ellipse[0]
        angle = -1 * math.radians(ellipse[2])
        curr_fit = 0
        # From http://answers.opencv.org/question/20521/how-do-i-get-the-goodness-of-fit-for-the-result-of-fitellipse/
        # rotates point to ellipse basis to get distance - avoids transcendental problem
        for point in hull:
            rot_x = (point[0][0] - center[0]) * numpy.cos(angle) - (point[0][1] - center[1]) * numpy.sin(angle)
            rot_y = (point[0][0] - center[0]) * numpy.sin(angle) + (point[0][1] - center[1]) * numpy.cos(angle)
            curr_fit += abs((rot_x / ellipse[1][0]) ** 2 + (rot_y / ellipse[1][1]) ** 2 - 0.25)
        if curr_fit < fit_number:
            fit_number = curr_fit
            best_index = i
        print("ellipse:", ellipse)
        print("ratio:", curr_fit)
    # cv2.imwrite("output/trash.png", trash)

    return {
            "turn": math.degrees(get_turning_angle(ellipses[best_index][0][0])),
            "distance": kinect_depth_to_meters(depth_at_pixel(depth, ellipses[best_index][0]))
           }


def main():
    image, depth = read_kinect_image(ir=True)
    rgb = read_kinect_image()
    cv2.imwrite("output/ir.png", image)
    cv2.imwrite("output/rgb" + str(int(time.time()/1)) + ".png", rgb)
    cv2.imwrite("output/depth.png", pretty_depth_cv(numpy.copy(depth)))
    numpy.save("depth.npy", depth)
    print(depth)
    # position = get_peg_info(image, depth)
    # print("position:", position)
    print(get_gear_info(rgb, depth))

if __name__ == '__main__':
    main()
