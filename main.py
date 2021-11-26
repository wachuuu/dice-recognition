import cv2
import numpy as np


def load_picture(filename):
    path = "./assets/{}.jpg".format(filename)
    # returns two arrays: original picture and grayscale values from 0 to 255
    return cv2.imread(path), cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def show_picture(img, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, len(img[0]), len(img))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(img):
    # blurring picture to reduce noise
    img = cv2.blur(img, (3, 3))
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    img = cv2.Canny(img, 80, 230)

    return img


def dice_detection(img, img_color):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dice_rects = []
    counter = 0

    for i, c in enumerate(contours):
        # Output tuple:
        # center(x,y) - rectangle mass centre,
        # size(width, height),
        # the rotation angle in clockwise direction
        dr_temp = cv2.minAreaRect(c)

        if dr_temp[1][1] != 0:  # sometimes height is equal to zero
            aspect_ratio = abs((dr_temp[1][0] / dr_temp[1][1]) - 1)
        else:
            aspect_ratio = 1
        area = dr_temp[1][0] * dr_temp[1][1]

        # filtering too big size or wrong shape rectangles
        if aspect_ratio < 0.25 and 2000 < area < 4000:
            process = True

            # finding duplicate dice contours
            for dice in dice_rects:
                if abs(dr_temp[0][0] - dice[0][0]) < 10 and abs(dr_temp[0][1] - dice[0][1]) < 10:
                    process = False
                    break

            if process:
                counter += 1
                dice_rects.append(dr_temp)

                # drawing the contour on the base image
                cnt_len = cv2.arcLength(c, True)
                cnt = cv2.approxPolyDP(c, 0.05 * cnt_len, True)
                for j in range(4):
                    cv2.line(img_color, cnt[j][0], cnt[(j + 1) % 4][0], (0, 0, 255), 2)

    print(f"{counter} dice detected")
    cv2.putText(img_color, f"DICE {counter}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    show_picture(img_color, 'Dice detection')

    return dice_rects


def main():
    img_color, img_gray = load_picture("testpicture")
    img_gray = preprocess(img_gray)
    dice_rects = dice_detection(img_gray, img_color)
    # show_picture(img_color)


if __name__ == "__main__":
    main()
