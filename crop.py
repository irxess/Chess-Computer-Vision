import cv2 as cv
import numpy as np

def transform(image, corners, square_size):
    transformed_size = square_size * 8 # 8 chess squares
    s = transformed_size
    transformed_corners = np.array([[0,0],[s,0],[0,s],[s,s]], dtype="float32")
    print(transformed_corners)
    transform_matrix = cv.getPerspectiveTransform(corners,transformed_corners)
    result = cv.warpPerspective(image,transform_matrix,(s,s))

    print('Transformed the image.')
    cv.imwrite('images/' + 'transform' + '.png', result)

    return result

def cut_and_save(image, x_size, y_size):
    c_x = np.arange(0,x_size*8+1,x_size)
    c_y = np.arange(0,y_size*8+1,y_size)

    for x in range(8):
        for y in range(8):
            print(c_x[x])
            print(c_x[x+1])
            print('/////')
            print(c_y[y])
            print(c_y[y+1])
            print('/////')
            print(image.shape)
            square = image[c_x[x]:c_x[x+1], c_y[y]:c_y[y+1]]
            cv.imwrite('images/squares/' + str(x) + str(y) + '.png', square)
