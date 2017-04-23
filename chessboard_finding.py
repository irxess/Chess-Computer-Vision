import cv2 as cv
import numpy as np
from astar import Node, Graph, AStar
import good_features as gf

def find_chessboard(image, method):
    if method == 'greendiff':
        green = greendiff(image)
        corners = find_corners_using_astar(green)
    elif method == 'goodfeatures':
        all_corners = gf.find_81_corners(image)
        gf.draw_corners(all_corners, image, 'all_corners')
        lines = gf.find_lines_from_points( all_corners )
        g1,g2 = gf.find_largest_line_groups(lines)
        chess_board_lines = gf.find_outer_lines(g1)
        chess_board_lines += gf.find_outer_lines(g2)
        corners = gf.find_intersections(chess_board_lines)
        gf.draw_corners(corners, image, 'board_corners')

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

# assume that the middle pixel of the image is part of the chess board
def find_corners_using_astar(image):
    x,y,_ = image.shape

    corners = np.empty((4, 2), dtype="float32")
    i = 0
    for c in [(0,0),(x-1,0),(0,y-1),(x-1,y-1)]:
        corner = find_astar_corner(image,c[0],c[1])
        corners[i][0] = corner[0]
        corners[i][1] = corner[1]
        i += 1
        print('Found the coordinates of a corner.')
    return corners


def find_astar_corner(image, corner_x, corner_y):
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
