import cv2 as cv
import numpy as np


def find_chessboard(image, method):
    if method == 'greendiff':
        greendiff(image)
        # Do stuffs (A*?) to find corners
    # Add different methods here
    else:
        print("Method not implemented or not found")
        return

    cv.imwrite('images/' + method + '.png',image)
    # return coordinates

def greendiff(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    std_colors = image.std(axis=2)
    std_threshold = (std_colors.max().max() + std_colors.mean().mean()) / 2

    green = image[:,:,1]
    max_green = green.max(axis=0).max()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x,y,0] = 0
            image[x,y,2] = 0
            if image[x,y,1] > gray[x,y] and std_colors[x,y] > std_threshold:
                    image[x,y,1] = max_green
            else:
                image[x,y,1] = 0
