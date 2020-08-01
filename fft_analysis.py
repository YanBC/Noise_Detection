import cv2 as cv
import numpy as np
import os


def show_freq(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(f))
    magnitude_spectrum -= magnitude_spectrum.min()
    magnitude_spectrum *= 255.0 / magnitude_spectrum.max()
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

    cv.imshow('original', img)
    cv.waitKey()

    cv.imshow('freq', magnitude_spectrum)
    cv.waitKey()

    cv.destroyAllWindows()


if __name__ == '__main__':
    imageDir = './images'
    imageFiles = os.listdir(imageDir)

    for imageFile in imageFiles:
        imagePath = os.path.join(imageDir, imageFile)
        image = cv.imread(imagePath, 0)

        show_freq(image)