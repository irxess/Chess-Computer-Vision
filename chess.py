import cv2 as cv
import os
import chess_deep
from chessboard_finding import find_chessboard
from crop import transform, cut_and_save


def main(name):
    image = open_image(name)
    coordinates = find_chessboard(image, "greendiff")
    chessboard_image = transform(image, coordinates, 50)
    cut_and_save(chessboard_image, 50, 50)

    # Load squares

    # Check for model
    if not os.path.isfile("/tmp/chess_model.ckpt"):
        #train
        pass

    # classify

    # translate to human readable output

    # Something like this, depending on method:
    # coordinates = find_chessboard(image)
    # chess_board = transform(image, coordinates)
    # chess_squares = magic(image)
    # chess_pieces = more_magic(chess_squares)
    # print(chess_pieces)


def open_image(name):
    path = os.path.join(os.getcwd(), "images/", name + ".png")
    if os.path.exists(path):
        image = cv.imread(path)
    else:
        print("The path " + path + " could not be found.")
        return
    return image

def read_square_images():
    sq_dir = os.path.join(os.getcwd(), "images", "squares")
    #for filname in os.listdir(sq_dir):
        #if filname[0] in
    pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "board1"
    main(image_path)
