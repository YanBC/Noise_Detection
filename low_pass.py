import cv2 as cv
import numpy as np
import os
from scipy.ndimage.filters import convolve

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    energy = np.sum(convolved)

    return energy


def mse(img1, img2):
    error = (abs(img1 - img2)).mean()
    return error

def diff_energy(img1, img2):
    energy1 = calc_energy(img1)
    energy2 = calc_energy(img2)

    return abs(energy1 - energy2)


# def isNoisy(img, thres=0.2):
#     blur = cv.medianBlur(img, 3)

#     n = diff(blur, img)
#     d = img.mean()

#     if n / d > thres:
#         return true
#     else:
#         return false


if __name__ == '__main__':
    imageDir = './images'
    imageFiles = os.listdir(imageDir)

    for imageFile in imageFiles:
        imagePath = os.path.join(imageDir, imageFile)
        image = cv.imread(imagePath)
        image_h, image_w, image_c = image.shape

        blur = cv.medianBlur(image, 3)
        n = diff_energy(blur, image) / (image_h * image_w * image_c)
        noise_ratio = n

        print(imagePath, noise_ratio)

        original_windowName = f'original: {imagePath}'
        blur_windowName = f'blur: {noise_ratio}'

        cv.namedWindow(original_windowName, cv.WINDOW_NORMAL)
        cv.imshow(original_windowName, image)
        cv.waitKey()
        cv.namedWindow(blur_windowName, cv.WINDOW_NORMAL)
        cv.imshow(blur_windowName, blur)
        cv.waitKey()

        cv.destroyAllWindows()