#!/usr/bin/env python3
"""
Given an IR image and a depth array, calculates angles to turn and distance to lift for FRC Steamworks 2017.
Also, given an RGB image and a depth array, calculates angles to turn and distance to gear.

Parth Oza
Jagan Prem
Oksana Tkach
Some code originally generated using GRIP
FRC Team 4099
"""

import cv2
import numpy
import math
from itertools import combinations

DISTANCE_LIFT_TAPE_METERS = 0.26  # outside edges
FOV_OF_CAMERA = math.radians(57)
IMAGE_WIDTH_PX = 640


class TooMuchInterferenceException(Exception):
    """
    The Exception raised when there is too much stuff in an image to positively identify target
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LiftNotFoundException(Exception):
    """
    The Exception raised when there is no goal in the image
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class GearNotFoundException(Exception):
    """
    The Exception raised when there is no gear in the image
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def blur_image(src, radius):
    """
    Uses OpenCV blur function to blur src given blur radius - eliminates most noise in image
    :param src: Source image to blur
    :param radius: blur radius
    :return: blurred image
    """
    ksize = int(6 * round(radius) + 1)
    return cv2.blur(src, (ksize, ksize), round(radius))


def lift_binary_threshold(src):
    """
    Uses OpenCV binary threshold to zero pixels below certain value and white the rest. Works on grayscale images.
    Used for locating LIFT on AIRSHIP given a blurred IR image.
    :param src: image to threshold (grayscale, usually IR image)
    :return: thresholded image
    """
    return cv2.threshold(src, 30, 255, cv2.THRESH_BINARY)[1]


def gear_hsv_threshold(src):
    """
    Uses HSV thresholding to locate specific shade of yellow - checks for pixel values in given range, blacks out
    everything not in that range. Used to locate GEAR on ground.
    :param src: source RGB image
    :return: thresholded image
    """
    hue, sat, val = [(11, 180), (170, 255), (124, 255)]
    out = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))


def get_contours(src):
    """
    Finds a set of contours (borders of blocks of same color) in image. Usually used on thresholded image to locate
    blocks of things of one color.
    :param src: image to find contours in (usually thresholded image)
    :return: array of contours (each contour is a list of points on the outside of blocks of color)
    """
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(src, mode=mode, method=method)
    # print("len of contours:", len(contours))
    return contours


def lift_filter_contours(input_contours):
    """
    Filters contours given to eliminate those which are definitely not reflective tape on the LIFT
    :param input_contours: list of contours
    :return: list of contours, without the ones that don't fit LIFT criteria
    """
    output = []
    for contour in input_contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        if len(contour) < 0 or len(contour) > 200:
            # print("too many sides :/", len(contour))
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        if ratio < 0 or ratio > 0.7:
            # print("ur ratio is trash", ratio)
            continue
        output.append(contour)
    return output


def y_filter_rectangles(rectangles, offset=100):
    """
    Removes rectangles from list which are farther than given offset from the median of the y positions of rectangles
    :param rectangles: list of rectangles to start
    :param offset: number of pixels off from median to be to discard rectangle
    :return: filtered list of rectangles
    """
    median_y = numpy.median([rectangle[1] for rectangle in rectangles])
    return list(filter(lambda rectangle: abs(rectangle[1] - median_y) <= offset, rectangles))


def distance_squared(p1, p2):
    """
    Returns the squared distance between two points given x and y coordinates. Skips square root to save time - not
    necessary for simply comparing distances since if squared distance is greater, distance must be greater
    :param p1: point 1 of segment to get distance for
    :param p2: point 2 of segment to get distance for
    :return: the distance between p1 and p2 squared
    """
    print("p1, p2:", p1, p2)
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))


def get_lift_position_from_src(src):
    """
    Given a source image, gets the position of two reflective tapes (the LIFT). Uses above functions to achieve this.
    Pairing is done if there are 3 or 4 pieces of reflective tape in the image - that means 2 sides of the AIRSHIP are
    seen. This is done by similarity of apparent size.
    :param src: the source IR image to find the LIFT in
    :return: two coordinates - one for each piece of reflective tape - left outside center, right outside center
    """
    blur = blur_image(src, 5)

    threshold = lift_binary_threshold(blur)
    cv2.imwrite("output/threshold.png", threshold)

    contours = get_contours(threshold)
    filtered_contours = lift_filter_contours(contours)
    # contour_drawn = cv2.cvtColor(numpy.copy(threshold), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(contour_drawn, filtered_contours, -1, (0,255,0), 3)
    # cv2.imwrite("output/contoured.png", contour_drawn)
 
    rectangles = []
    
    for contour in filtered_contours:
        rectangles.append(cv2.boundingRect(contour))

    rectangles = y_filter_rectangles(rectangles)

    if len(rectangles) < 2:
        raise LiftNotFoundException("yo no goals")
    elif len(rectangles) > 4:
        raise TooMuchInterferenceException("why is everything reflective :/")

    rectangles.sort()

    if len(rectangles) == 3:
        combos = list(combinations((0, 1, 2), 2))
        pair = min((0, 1, 2), key=lambda i: distance_squared(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        rectangles = [rectangles[combos[pair][0]], rectangles[combos[pair][1]]]
    elif len(rectangles) == 4:
        combos = list(combinations((0, 1, 2, 3), 2))
        pair1 = min(range(len(combos)), key=lambda i: distance_squared(rectangles[combos[i][0]][:2], rectangles[combos[i][1]][:2]))
        pair2 = len(combos) - pair1 - 1

        def size(i): return rectangles[i][2] * rectangles[i][3]

        if size(combos[pair1][0]) + size(combos[pair1][1]) > size(combos[pair2][0]) + size(combos[pair2][1]):
            rectangles = [rectangles[combos[pair1][0]], rectangles[combos[pair1][1]]]
        else:
            rectangles = [rectangles[combos[pair2][0]], rectangles[combos[pair2][1]]]

    corner_set1, corner_set2 = rectangles[0], rectangles[1]

    final_coordinates = [[int(corner_set1[0]), int(corner_set1[1] + corner_set1[3] / 2)],
                         [int(corner_set2[0] + corner_set2[2]), int(corner_set2[1] + corner_set2[3] / 2)]]

    return final_coordinates


def get_turning_angle(center_x):
    """
    Gets the angle to turn to point at the LIFT given the center of the two pieces of reflective tape
    :param center_x: the X coordinate of the center of the LIFT. Uses camera FOV to calculate this.
    :return: the angle to turn to point at the LIFT
    """
    return angle_at_x(center_x) - FOV_OF_CAMERA / 2


def get_offset_angle(distance_1, distance_2, angle_1, angle_2):
    """
    Gets the angle from horizontal that the LIFT is - useful to see how off you are from running into the AIRSHIP
    properly.
    :param distance_1: distance to left reflective tape
    :param distance_2: distance to right reflective tape
    :param angle_1: angle from left side of camera to left reflective tape
    :param angle_2: angle from left side of camera to right reflective tape
    :return: the angle from horizontal the LIFT is at
    """
    f = (FOV_OF_CAMERA / 2)
    return numpy.arcsin((distance_2 - (distance_1 * numpy.cos(f - angle_1) / numpy.cos(f - angle_2))) * numpy.cos(f - angle_2) / DISTANCE_LIFT_TAPE_METERS)


def kinect_depth_to_meters(kinect_depth):
    """
    Converts from raw data value for Kinect depth (0-1023) to distance in meters. Calculated using a inverse linear
    regression with data points collected manually using our specific Kinect. Apparently there is variation for each
    Kinect - may have to calculate this again for a different sensor. Take the inverse of each real depth value and
    then do linear regression on that to get an equation in this form (y = 1 / (a + bx))
    :param kinect_depth: the value from 0 to 1023 that the Kinect provides
    :return: the distance in meters corresponding to that value
    """
    return 1 / (3.14674327 + -0.0028990952 * kinect_depth)


def depth_at_pixel(depth_array, pixel, vertical_step=1, side_step=-1):
    """
    Calculates the distance in meters to a pixel on the IR or RGB image. If the depth array shows 0 for that pixel, it
    goes up that column to find a nonzero value. If none is found, it will go left until a nonzero value.
    :param depth_array: Frame of depth data produced by the Kinect
    :param pixel: Pixel coordinates to get distance to
    :param vertical_step: The distance to go vertically to avoid zero values
    :param side_step: The distance to go horizontally to avoid zero values
    :return: Distance to given pixel in meters
    """
    new_pixel = numpy.copy(pixel)
    while 0 < new_pixel[1] < depth_array.shape[0] - 1 and depth_array[int(new_pixel[1]), int(new_pixel[0])] == 0:
        new_pixel[1] += vertical_step
    if depth_array[int(new_pixel[1])][int(new_pixel[0])] == 0:
        new_pixel = numpy.copy(pixel)
        while 0 < new_pixel[0] < depth_array.shape[1] - 1 and depth_array[int(new_pixel[1]), int(new_pixel[0])] == 0:
            new_pixel[0] += side_step
    # print("raw depth:", depth_array[new_pixel[1], new_pixel[0]])
    return kinect_depth_to_meters(depth_array[int(new_pixel[1]), int(new_pixel[0])])


def angle_at_x(x_value):
    """
    Calculates the angle from the leftmost part of the FOV of the Kinect to the given x value in an image
    :param x_value: the x value to get the angle to
    :return: the angle from the left of the FOV to the x coordinate given
    """
    return (x_value / IMAGE_WIDTH_PX) * FOV_OF_CAMERA


def get_lift_info(image, depth):
    """
    Calculates offset angle, turning angle, and distance to LIFT given IR image and depth data
    :param image: Kinect IR image containing 2 pieces of reflective tape
    :param depth: Kinect depth frame corresponding to image
    :return: dictionary containing keys for "offset" and "turn" (both in degrees) and "distance" in meters.
    """
    # cv2.imwrite("output/ir.png", image)
    # cv2.imwrite("output/depth.png", pretty_depth_cv(numpy.copy(depth)))
    position = get_lift_position_from_src(image)
    depth_at_1 = depth_at_pixel(depth, position[0], side_step=1)
    depth_at_2 = depth_at_pixel(depth, position[1])
    angle_at_1 = angle_at_x(position[0][0])
    angle_at_2 = angle_at_x(position[1][0])
    offset_angle = math.degrees(get_offset_angle(depth_at_1, depth_at_2, angle_at_1, angle_at_2))
    turning_angle = math.degrees(get_turning_angle((position[0][0] + position[1][0]) / 2))
    return {"offset": offset_angle, "turn": turning_angle, "distance": (depth_at_1 + depth_at_2) / 2}


def get_gear_info(image, depth):
    """
    Calculates turning angle and distance to gear given RGB image and depth data
    :param image: RGB image from Kinect containing gear
    :param depth: depth data corresponding to image
    :return: dictionary containing keys for "turn" in degrees and "distance" in meters
    """
    blur = blur_image(image, 5)
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
        # print("ellipse:", ellipse)
        # print("ratio:", curr_fit)
    # cv2.imwrite("output/trash.png", trash)
    return {
            "turn": math.degrees(get_turning_angle(ellipses[best_index][0][0])),
            "distance": kinect_depth_to_meters(depth_at_pixel(depth, ellipses[best_index][0]))
           }


def main():
    import streamer
    rgb, image, depth = streamer.read_kinect_images()
    # cv2.imwrite("output/ir.png", image)
    # cv2.imwrite("output/rgb" + str(int(time.time()/1)) + ".png", rgb)
    # cv2.imwrite("output/depth.png", pretty_depth_cv(numpy.copy(depth)))
    # numpy.save("depth.npy", depth)
    depth = numpy.load("depth.npy")
    # image = cv2.cvtColor(cv2.imread("output/ir.png"), cv2.COLOR_BGR2GRAY)
    # print(depth)
    # rgb = cv2.imread("output/rgb1486163995.png")
    print("lift:", get_lift_info(image, depth))
    print("gear:", get_gear_info(rgb, depth))

if __name__ == "__main__":
    main()
