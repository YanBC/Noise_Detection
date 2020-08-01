import cv2 as cv
import numpy as np
import os


def diff(img1, img2):
    mse = (np.absolute(img1 - img2)).mean()
    return mse

def laplacian_trans(img):
    laplacian64f = cv.Laplacian(img, cv.CV_64F)
    abs_laplacian64f = np.absolute(laplacian64f)
    laplacian8u = np.uint8(abs_laplacian64f)

    return laplacian8u


if __name__ == '__main__':
    imageDir = './images'
    imageFiles = os.listdir(imageDir)

    for imageFile in imageFiles:
        imagePath = os.path.join(imageDir, imageFile)
        image = cv.imread(imagePath)

        laplace_image = laplacian_trans(image)

        n = diff(laplace_image, image)
        noise_ratio = n

        print(imagePath, noise_ratio)

        original_windowName = f'original: {imagePath}'
        laplacian_windowName = f'laplacian: {noise_ratio}'

        cv.namedWindow(original_windowName, cv.WINDOW_NORMAL)
        cv.imshow(original_windowName, image)
        cv.waitKey()
        cv.namedWindow(laplacian_windowName, cv.WINDOW_NORMAL)
        cv.imshow(laplacian_windowName, laplace_image)
        cv.waitKey()

        cv.destroyAllWindows()