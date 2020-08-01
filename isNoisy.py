import cv2 as cv
import numpy as np


MAX_DIFF = 50


def e1_opencv(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grads_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grads_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    grads = abs(grads_x) + abs(grads_y)

    return grads


def diff_energy(img1, img2):
    assert img1.shape == img2.shape
    energy1 = e1_opencv(img1)
    energy2 = e1_opencv(img2)
    image_h, image_w, image_c = img1.shape

    sub = np.sum(energy1 - energy2) / (image_h * image_w * image_c)

    return sub


def noise_level(image):
    processed = cv.medianBlur(image, 3)
    diff = diff_energy(image, processed)

    if diff > MAX_DIFF:
        diff = MAX_DIFF

    level = diff / MAX_DIFF
    return level


def isnoisy(image, thres=0.6):
    confidence = noise_level(image)

    if confidence > thres:
        return True
    else:
        return False



if __name__ == '__main__':
    import os

    imageDir = './images'
    imageFiles = os.listdir(imageDir)
    cv.namedWindow('show', cv.WINDOW_NORMAL)

    for imageFile in imageFiles:
        imagePath = os.path.join(imageDir, imageFile)
        image = cv.imread(imagePath)

        # print(isnoisy(image))
        print(noise_level(image))

        cv.imshow('show', image)
        ch = cv.waitKey()
        if chr(ch) == 'q':
            break