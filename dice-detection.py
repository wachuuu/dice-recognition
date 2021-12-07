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

        if dr_temp[1][0] != 0 and dr_temp[1][1] != 0:  # sometimes height is equal to zero
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

    print(f"{counter} - dice detected\n")
    # Putting dice count on base image
    cv2.putText(img_color, f"DICE {counter}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    return dice_rects


def dots_detection(dice_rects, img, img_color):
    dice_counts = [0, 0, 0, 0, 0, 0]

    for dr in dice_rects:
        rot_mat = cv2.getRotationMatrix2D(dr[0], dr[2], 1.0)
        rot_img = cv2.warpAffine(img, rot_mat, (len(img[0]), len(img)), cv2.INTER_CUBIC)
        crop_img = cv2.getRectSubPix(rot_img, (int(dr[1][0]) - 10, int(dr[1][1]) - 10), dr[0])

        ret, crop_img = cv2.threshold(crop_img, 130, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        dots_rects = []
        for i, c in enumerate(contours):
            d_temp = cv2.minAreaRect(c)

            if d_temp[1][0] != 0 and d_temp[1][1] != 0:
                aspect_ratio = abs((d_temp[1][0] / d_temp[1][1]) - 1)
            else:
                aspect_ratio = 1
            area = d_temp[1][0] * d_temp[1][1]

            if aspect_ratio < 0.4 and 8 < area < 150:

                process = True
                for dot in dots_rects:
                    if abs(d_temp[0][0] - dot[0][0]) < 10 and abs(d_temp[0][1] - dot[0][1]) < 10:
                        process = False
                        break

                if process:
                    dots_rects.append(d_temp)

                    cnt_len = cv2.arcLength(c, True)
                    cnt = cv2.approxPolyDP(c, 0.05 * cnt_len, True)
                    for j in range(4):
                        cv2.line(crop_img, cnt[j][0], cnt[(j + 1) % 4][0], (0, 0, 255), 2)

        print(f"{len(dots_rects)}: dots detected")
        if 1 <= len(dots_rects) <= 6:
            dice_counts[len(dots_rects) - 1] += 1

        # show_picture(crop_img, 'test')

    s = 0
    for i, d in enumerate(dice_counts):
        s += d
        cv2.putText(img_color, f"{i + 1}: {d}", (20, 55 + 25 * i), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1,
                    cv2.LINE_AA)

    print(f'{s} - sum of dices after dots counting\n')
    return dice_counts


def main():
    img_color, img_gray = load_picture("testpicture")
    img_gray_pp = preprocess(img_gray)

    dice_rects = dice_detection(img_gray_pp, img_color)
    dice_counts = dots_detection(dice_rects, img_gray, img_color)

    show_picture(img_color, 'Dice recognition')


if __name__ == "__main__":
    main()
