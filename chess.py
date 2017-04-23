import cv2 as cv
from skimage import io
import numpy as np
import os
import chess_deep
from chessboard_finding import find_chessboard
from crop import transform, cut_and_save


def main(name):
    image = open_image(name)
#    coordinates = find_chessboard(image, "greendiff")
    coordinates = find_chessboard(image, "goodfeatures")
    chessboard_image = transform(image, coordinates, 50)
    cut_and_save(chessboard_image, 50, 50)

    # Check for model
    if not os.path.isfile("/tmp/checkpoint"):
        chess_deep.train_model()

    # Load squares
    chess_squares = read_square_images()

    # Classify
    chess_squares = chess_deep.classify_squares(chess_squares)

    # translate to human readable output
    print_classes(chess_squares)


def open_image(name):
    path = os.path.join(os.getcwd(), "images/", name + ".jpg")
    if os.path.exists(path):
        image = cv.imread(path)
    else:
        print("The path " + path + " could not be found.")
        return
    return image


def read_square_images():
    squares = []
    sq_dir = os.path.join(os.getcwd(), "images", "squares")
    for i in range(8):
        for j in range(8):
            filename = str(i) + str(j) + ".png"
            img = io.imread(os.path.join(sq_dir, filename))[1:49, 1:49]
            squares.append(img)
    return np.array(squares)


def class_to_letter(c):
    letters = [' ', 'K', 'Q', 'R', 'N', 'B', 'P', 'k', 'q', 'r', 'n', 'b', 'p']
    return letters[c]


def print_classes(classes):
    for i in range(8):
        for j in range(8):
            print('[' + class_to_letter(classes[i+j*8]) + ']', end=' ')
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "board0"
    main(image_path)
