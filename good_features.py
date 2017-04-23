import numpy as np
import cv2
from math import sqrt, atan
from copy import deepcopy
from operator import itemgetter
#from matplotlib import pyplot as plt

img = cv2.imread('images/board1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#blurred = cv2.GaussianBlur(gray, (55,55), 0)
#blurred = cv2.blur(gray, (60,60))
blurred = cv2.medianBlur(gray, 31)
#blurred = cv2.bilateralFilter(gray, 9,75,75)

corners = cv2.goodFeaturesToTrack(blurred,81,0.01,170)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),25,255,-1)

cv2.imwrite('images/test.png',img)

#plt.imshow(img),plt.show()

def create_line(a, b):
    ax, ay = a[0]
    bx, by = b[0]
    return (ax,ay,bx,by)

def line_to_point_dist(l, p):
    x0,y0 = p[0]
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
a_points = corners.tolist()
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

# group lines by similair angle
# assume that the groups we're interested in are the largest
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
        if len(new_list) > len(most_lines):
            next_most_lines = most_lines
            most_lines = new_list
        elif len(new_list) > len(next_most_lines):
            next_most_lines = new_list
last_list = sorted_lines[prev_index:]
all_lists.append(last_list)
if len(last_list) > len(most_lines):
    next_most_lines = most_lines
    most_lines = last_list
elif len(last_list) > len(next_most_lines):
    next_most_lines = last_list


# find the outmost lines
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


chess_board_lines = find_outer_lines(most_lines)
chess_board_lines += find_outer_lines(next_most_lines)

# for line in chess_board_lines:
#     color = (0,255,0) # green
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),color,5)

for l in all_lists:
    for line in l:
        angle = line[1]
        x1,y1,x2,y2 = line[0]
        color = (255,240,0) # cyan
        cv2.line(img,(x1,y1),(x2,y2),color,5)

for line in most_lines:
    angle = line[1]
    x1,y1,x2,y2 = line[0]
    color = (0,255,0) # green
    cv2.line(img,(x1,y1),(x2,y2),color,5)

for line in next_most_lines:
    angle = line[1]
    x1,y1,x2,y2 = line[0]
    color = (0,0,255) # red
    cv2.line(img,(x1,y1),(x2,y2),color,5)

# for l in lines:
#     s = l[1]
#     if s < 0.01 and s > 0:
#         color = (255,111,255) # purple
#     elif s < -0.01 and s > -0.06:
#         color = (0,255,0) # green
#     else:
#         color = (0,0,255) # red
#     x1,y1,x2,y2 = l[0]
#     cv2.line(img,(x1,y1),(x2,y2),color,5)
cv2.imwrite('images/test2.png',img)


# Images
# Board0
# -89 : -89
# -47 : -45
# -2 : 1
# 26
# 42 : 44
# 86 : 89
