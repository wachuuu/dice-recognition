import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

RESIZED_WIDTH = 700

def load_picture(filename):
    path = "./assets/{}.jpg".format(filename)
    img = cv2.imread(path)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # resize picture to {RESIZED_WIDTH} px and keep original ratio
    width = int(img.shape[1])
    height = int(img.shape[0])
    resize_ratio =  RESIZED_WIDTH / width
    resized_height = int(height * resize_ratio)

    resized = cv2.resize(img, (RESIZED_WIDTH, resized_height), interpolation=cv2.INTER_AREA)
    resized_gray = cv2.resize(img_gray, (RESIZED_WIDTH, resized_height), interpolation=cv2.INTER_AREA)

    # returns two arrays: original picture and grayscale values from 0 to 255
    return resized, resized_gray


def show_picture(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, len(img[0]), len(img))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_picture(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def main():
    img_color, img_gray = load_picture("dice_tally")

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    # dot min and max areas
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1000
    # how round blob should be to be considered as a dot
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # reject tilted or deformed dots
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    # image thersholds
    params.minThreshold = 10 
    params.maxThreshold = 200
    dark_dots_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = dark_dots_detector.detect(img_gray)
    # detect white dots on a dark dice
    params.blobColor = 255
    white_dots_detector = cv2.SimpleBlobDetector_create(params)
    keypoints += white_dots_detector.detect(img_gray)

    img_with_keypoints = cv2.drawKeypoints(img_color, keypoints, np.array([]), (255, 10, 100), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_picture(img_with_keypoints, 'Dice recognition')

if __name__ == "__main__":
    main()
