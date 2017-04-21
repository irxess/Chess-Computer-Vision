import cv2 as cv
import os
from chessboard_finding import find_chessboard
from crop import transform, cut_and_save

def main(name):
    image = open_image(name)
    coordinates = find_chessboard(image, "greendiff")
    chessboard_image = transform(image, coordinates, 50)
    cut_and_save(chessboard_image, 50, 50)
    # Something like this, depending on method:
    # coordinates = find_chessboard(image)
    # chess_board = transform(image, coordinates)
    # chess_squares = magic(image)
    # chess_pieces = more_magic(chess_squares)
    # print(chess_pieces)

def open_image(name):
    path = os.path.join(os.getcwd(), "images/", name + ".jpg")
    if os.path.exists(path):
        image = cv.imread(path)
    else:
        print("The path " + path + " could not be found.")
        return
    return image


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "board0"
    main(image_path)
