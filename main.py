#!/usr/bin/env python3
"""
Code originally from GRIP
"""

DISTANCE_LIFT_TAPE = 10.25	#outside edges

def gaussian_blur(src):
    radius = 3
	ksize = int(6 * round(radius) + 1)
    return cv2.GaussianBlur(src, (ksize, ksize), round(radius))

def hsv_threshold(src):
    hue, sat, val = [(0, 180), (0, 0), (128, 255)]
	out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

def get_contours(src):
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
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
        ratio = (float)(w) / h
        if ratio < 0 or ratio > 1000:
            continue
        output.append(contour)
    return output

def hulls(input_contours):
    return [cv2.convexHull(contour) for contour in output]

def get_position_from_src(src):
    hull_position = hulls(filter_contours(get_contours(hsv_threshold(blur(src)))))

def find_turning_angle(to_target_edge, left_distance, right_distance):
	"""
	Params:
	to_target_edget = angle from front of robot to the nearest tape
	left_distance and right_distance = distance from robot to left/right tape
	"""
    sides = (left_distance ** 2 + right_distance ** 2 - DISTANCE_LIFT_TAPE ** 2) / 2 * left_distance * right_distance
    theta = numpy.arccos(sides)
    return to_target_edge + theta/2

def main():
    
    pass

if __name__ == '__main__':
	main()
