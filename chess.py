import cv2 as cv
from chessboard_finding import find_chessboard

def main(name):
    image = open_image(name)
    find_chessboard(image, "greendiff")
    # Something like this, depending on method:
    # coordinates = find_chessboard(image)
    # chess_board = transform(image, coordinates)
    # chess_squares = magic(image)
    # chess_pieces = more_magic(chess_squares)
    # print(chess_pieces)

def open_image(name):
    path = "images/" + name + ".jpg"
    image = cv.imread(path)
    return image

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "board0"
    main(image_path)
