import cv2 as cv
import numpy as np
from astar import Node, Graph, AStar


def find_chessboard(image, method):
    if method == 'greendiff':
        green = greendiff(image)
        corners = find_corners_using_astar(green)
    # Add different methods here
    else:
        print("Method not implemented or not found")
        return

    return corners


def greendiff(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    std_colors = image.std(axis=2)
    std_threshold = (std_colors.max().max() + std_colors.mean().mean()) / 2

    green_border = np.empty([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            green_border[x,y,0] = 0
            green_border[x,y,2] = 0
            if image[x,y,1] > gray[x,y] and std_colors[x,y] > std_threshold:
                green_border[x,y,1] = 255
            else:
                green_border[x,y,1] = 0
    cv.imwrite('images/' + 'greendiff' + '.png', green_border)
    print("Filtered the green in the image.")
    return green_border


def find_corners_using_astar(image):
    # assume that the chess board is not rotated too much
    # assume that the middle pixel of the image is part of the chess board
    x,y,_ = image.shape

    corners = np.empty((4, 2), dtype="float32")
    i = 0
    for c in [(0,0),(x-1,0),(0,y-1),(x-1,y-1)]:
        corner = find_corner(image,c[0],c[1])
        corners[i][0] = corner[0]
        corners[i][1] = corner[1]
        i += 1
        print('Found the coordinates of a corner.')
    return corners


def find_corner(image, corner_x, corner_y):
    x,y,_ = image.shape
    g = Graph(x,y,image)
    g.startNode = g.grid[x//2][y//2]
    g.goalNode = g.grid[corner_x][corner_y]
    state = AStar(g)

    result = state.iterateAStar()
    best = state.bestNode
    updates = 0
    while result.state != 'goal' and updates < 50:
        result = state.iterateAStar()
        new_best = state.bestNode
        if new_best == best:
            updates += 1
        else:
            best = new_best
            updates = 0
    return best.y, best.x # yes, in this order

    # color the board
    # for x in range(image.shape[0]):
    #     for y in range(image.shape[1]):
    #         # ['start', 'goal', 'unvisited', 'closed', 'open', 'blocked', 'path']
    #         if g.grid[x][y].state == 'path':
    #             image[x,y,0] = 255
    #             image[x,y,2] = 255
    #         elif g.grid[x][y].state == 'closed':
    #             image[x,y,0] = 0
    #             image[x,y,1] = 0
    #             image[x,y,2] = 255
    #         elif g.grid[x][y].state == 'open':
    #             image[x,y,0] = 255
    #             image[x,y,1] = 255
    #             image[x,y,2] = 0
    #         elif g.grid[x][y].state == 'start':
    #             image[x,y,0] = 255
    #             image[x,y,2] = 255
