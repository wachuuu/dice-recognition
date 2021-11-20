import cv2


def load_picture(filename):
    path = "./assets/{}.jpg".format(filename)
    # returns two arrays: original picture and grayscale values from 0 to 255
    return cv2.imread(path), cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def show_picture(img):
    cv2.imshow("Dice recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(img):
    # blurring picture to reduce noise
    img = cv2.blur(img, (3, 3))
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    img = cv2.Canny(img, 80, 230)
    return img


def main():
    img_color, img_gray = load_picture("testpicture")
    show_picture(preprocess(img_gray))


if __name__ == "__main__":
    main()
