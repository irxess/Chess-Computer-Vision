import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple

# Load image
orgImg = cv2.imread('images/board0.jpg')
#orgImg = cv2.imread('images/IMG_0132.jpg')
#orgImg = cv2.imread('images/IMG_0140.jpg')

# Grayscale image
gray = cv2.cvtColor(orgImg,cv2.COLOR_BGR2GRAY)

# Smooth image - attept to remove noise
gaus = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection using laplacian
laplacian = cv2.Laplacian(gaus,cv2.CV_8U)

# Morphological transformation closing
kernel = np.ones((3,3),np.uint8)
laplacian = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, kernel)

# Threshold using adaptive thresholding
thres = cv2.adaptiveThreshold(laplacian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)

### Show edge image
##plt.imshow(thres, cmap = 'gray', interpolation = 'bicubic')
##plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
##plt.show()
##cv2.imwrite('Threshold.jpg', thres)

# Detect lines with Hough Lines Transform
# threshold found with brute force
threshold = 800
lines = cv2.HoughLines(thres,1,np.pi/180,threshold)

# Make and sort a list of the houghlines found
Vector = namedtuple('Vector', ['amplitude', 'radians'])

hlines = []
for x in lines:
    hlines.append(Vector(x[0][0],x[0][1]))

hlines = sorted(hlines, key=lambda x: x.radians)

def similar(a,b, rad_diff, amp_diff): # Defines weither lines are close or not
    rad = min(abs(a.radians-b.radians), abs(abs(a.radians-b.radians)-2*np.pi))
    if rad < rad_diff and abs(a.amplitude-b.amplitude) < amp_diff:
        return True
    else:
        return False

def merge(temp): # Merges a list of close lines into one line
    amplitude = 0
    radians = 0
    # Set amplitude to the median
    temp = sorted(temp, key=lambda x: x.amplitude)
    amplitude = temp[len(temp)//2].amplitude
    # Set radians to the mean
    for x in temp:
        if x.radians < np.pi:
            radians += np.pi*2
        radians += x.radians
    radians = radians / len(temp)
    return Vector(amplitude, radians)

def mergeExcessLines(lines, rad_diff, amp_diff): # Compare all lines and merge close lines
    result = []
    while lines:
        a = lines.pop()
        temp = [a]
        for n in lines:
            if similar(a, n, rad_diff, amp_diff):
                temp.append(n)
                lines.remove(n)
        result.append(merge(temp))
    return result

# Attempt to merge close lines
# the values for rad and amp are brute forced
first_result = mergeExcessLines(hlines, 1.5, 100)
result = mergeExcessLines(first_result, 1.5, 100)

# Draw lines on gray image
gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
for x in result:
        a = np.cos(x.radians)
        b = np.sin(x.radians)
        x0 = a*x.amplitude
        y0 = b*x.amplitude
        x1 = int(x0 + 5000*(-b))
        y1 = int(y0 + 5000*(a))
        x2 = int(x0 - 5000*(-b))
        y2 = int(y0 - 5000*(a))

        cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),2)

# Show lines on gray image and print number of lines detected
print('Houghtransform detects: {} lines'.format(len(lines))) if(len(lines)>0) else print ('No lines found!')
print('After merging: {} lines'.format(len(result))) if(len(result)>0) else print ('No lines found!')
plt.imshow(gray, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
##cv2.imwrite('best-worstcase_images/Houghlines_worstcase.jpg', gray)
