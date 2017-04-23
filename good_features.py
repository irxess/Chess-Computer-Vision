import numpy as np
import cv2
from math import sqrt, atan
from copy import deepcopy
from operator import itemgetter


def find_81_corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #blurred = cv2.GaussianBlur(gray, (55,55), 0)
    #blurred = cv2.blur(gray, (60,60))
    blurred = cv2.medianBlur(gray, 31)
    #blurred = cv2.bilateralFilter(gray, 9,75,75)

    corners = cv2.goodFeaturesToTrack(blurred,81,0.01,170)
    weird_corners = np.int0(corners).tolist()
    corners = []
    for c in weird_corners:
        corners.append(c[0])
    return corners

def draw_corners(corners, img, filename):
    img_copy = img.copy()
    for i in corners:
        x,y = i
        cv2.circle(img_copy,(x,y),25,255,-1)
    cv2.imwrite('images/' + filename + '.png',img_copy)

def create_line(a, b):
    ax, ay = a
    bx, by = b
    return (ax,ay,bx,by)

def line_to_point_dist(l, p):
    x0,y0 = p
    x1,y1,x2,y2 = l
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = sqrt((y2-y1)**2 + (x2-x1)**2)
    return numerator/denominator

def find_slope(line):
    x1,y1,x2,y2 = line
    if x1 == x2:
        return 0
    if y1 == y2:
        return 999
    x_diff = (x1-x2)
    y_diff = (y1-y2)
    slope = y_diff/x_diff
    degrees = atan(slope) * 180 / np.pi
    return degrees

# Find lines that are close enough to multiple points
def find_lines_from_points(points):
    a_points = points
    result = []
    threshold = 30.0
    while a_points:
        a = a_points.pop()
        b_points = deepcopy(a_points) # copy a_points
        while b_points:
            b = b_points.pop()
            c_points = deepcopy(b_points) # copy b_points
            line = create_line(a,b)
            points_on_line = 2
            while c_points:
                c = c_points.pop()
                distance = line_to_point_dist(line, c)
                if distance < threshold:
                    points_on_line += 1
                    #a_points.remove(c)
                    b_points.remove(c)
            if points_on_line > 5:
                result.append(line)
    lines = []
    for r in result:
        s = find_slope(r)
        line = (r,s)
        lines.append(line)

    return lines

def update_line_groups(largest, next_largest, new):
    if len(new) > len(largest):
        next_largest = largest
        largest = new
    elif len(new) > len(next_largest):
        next_largest = new
    return largest, next_largest

# group lines by similair angle
# assume that the groups we're interested in are the largest
def find_largest_line_groups(lines):
    sorted_lines = sorted(lines,key=itemgetter(1))
    diffs = []
    most_lines = []
    next_most_lines = []
    all_lists = []
    prev_index = 0
    for i in range(len(sorted_lines)-1):
        diff = sorted_lines[i+1][1]-sorted_lines[i][1]
        if diff > 5:
            new_list = sorted_lines[prev_index:i+1]
            prev_index = i+1
            all_lists.append(new_list)
            most_lines, next_most_lines = update_line_groups(most_lines, next_most_lines, new_list)
    last_list = sorted_lines[prev_index:]
        #diff = last_list[-1][1] - abs(all_lists[0][0][1])
        #if diff < 5:
        #    print("Wrapping corners found.")
        #    all_lists[0] += last_list
        #else:
    all_lists.append(last_list)
    most_lines, next_most_lines = update_line_groups(most_lines, next_most_lines, last_list)
    return most_lines, next_most_lines

# find the outmost lines in a list of lines
def find_outer_lines(lines):
    angle = lines[0][1]
    if angle < 45 and angle > -45:
        axis = 1 # or is it 1?
    else:
        axis = 0
    lowest_value = 99999
    lowest_line = None
    highest_value = 0
    highest_line = None

    for line in lines:
        x1,y1,x2,y2 = line[0]
        avg = [(x1+x2)/2, (y1+y2)/2]
        if avg[axis] < lowest_value:
            lowest_line = line
            lowest_value = avg[axis]
        if avg[axis] > highest_value:
            highest_line = line
            highest_value = avg[axis]
    return [lowest_line, highest_line]

# for line in chess_board_lines:
#     color = (0,255,0) # green
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),color,5)
#
# cv2.imwrite('images/test2.png',img)

def find_line_intersection(pair, lines):
    i,j = pair
    x1,y1,x2,y2 = lines[i][0]
    x3,y3,x4,y4 = lines[j][0]
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    px_num = (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)
    px_den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    py_num = (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)
    py_den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    return (px_num//px_den, py_num//py_den)

def find_intersections(list_of_lines):
    lines_pairs = [(0,2), (1,2), (0,3), (1,3)]
    corners = []
    for pair in lines_pairs:
        corner = find_line_intersection(pair, list_of_lines)
        corners.append(corner)
        #cv2.circle(img,corner,40,(255,255,255),-1)
    #cv2.imwrite('images/test3.png',img)

    np_corners = np.empty((4, 2), dtype="float32")
    for i in range(len(corners)):
        np_corners[i][0] = corners[i][0]
        np_corners[i][1] = corners[i][1]
    return np_corners
