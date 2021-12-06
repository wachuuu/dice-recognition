import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


RESIZED_WIDTH = 700
class Dot:
    def __init__(self, x, y, diam):
        self.x = x
        self.y = y
        self.diam = diam


def get_assets():
    return [f for f in listdir("./assets/") if isfile(join("./assets/", f))]


def load_picture(filename):
    path = "./assets/{}".format(filename)
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


def plot_picture(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def calc_distance(current_dot, dots):
    for dot in dots:
        dot.dist = np.sqrt((current_dot.x - dot.x)**2 + (current_dot.y - dot.y)**2)
    dots.sort(key = lambda k: k.dist)
    return dots


def detect_dices(img, keypoints):
    result = [0] * 6
    i = 0
    dots = []               # all dots detected
    detection_queue = []    # dots for calculations 

    if (len(keypoints) == 0):
        return img, result

    for k in keypoints:
        img = cv2.circle(img, (int(k.pt[0]), int(k.pt[1])),  int(k.size/2), (0,255,0), 2)
        dots.append(Dot(k.pt[0], k.pt[1], k.size))
    detection_queue = dots[:]
    
    while (len(detection_queue) > 0 and len(dots) > 0 and i < 1000):
        current_dot = detection_queue.pop()
        detection_circle = 1.895 * current_dot.diam
        eps = 0.2 * current_dot.diam
        dots = calc_distance(current_dot, dots) # dots[0] = current_dot

        # [1]: nearest dot > 2*detection circle
        if (len(dots) == 1 or dots[1].dist > 2*(detection_circle+eps)):
            result[0] = result[0] + 1
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(2*(detection_circle+eps)), (255,0,0), 1)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam/2), (255,0,0), 2)

        # [2]: nearest dot in 2*detection_circle (+/- epsilon)
        elif (len(dots) > 1 and 2*(detection_circle-eps) < dots[1].dist < 2*(detection_circle+eps)):
            detection_queue = calc_distance(current_dot, detection_queue)
            next_dot = detection_queue.pop(0)
            result[1] = result[1] + 1
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(2*(detection_circle+eps)), (255,255,0), 1)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam/2), (255,255,0), 2)
            img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam/2), (255,255,0), 2)

        # [3] or [5]: two nearest dots in detection circle  
        elif (len(dots) > 2 and len(detection_queue) > 1 and (detection_circle-eps) < dots[1].dist < (detection_circle+eps) and (detection_circle-eps) < dots[2].dist < (detection_circle+eps)):
            # [5]: third nearest dot is also in detection circle (+/- epsilon)
            if (len(dots) > 4 and len(detection_queue) > 3 and (detection_circle-eps) < dots[3].dist < (detection_circle+eps)):
                detection_queue = calc_distance(current_dot, detection_queue)
                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(detection_circle+eps), (0,255,255), 1)
                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(current_dot.diam/2), (0,255,255), 2)
                for z in range(4):
                    if (len(detection_queue) > 0):
                        next_dot = detection_queue.pop(0)
                        img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)),  int(next_dot.diam/2), (0,255,255), 2)
                result[4] = result[4] + 1
            # [3]: third nearest dot outside detection circle
            elif (dots[3].dist > (detection_circle+eps)):
                detection_queue = calc_distance(current_dot, detection_queue)
                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(detection_circle+eps), (0,0,255), 1)
                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(current_dot.diam/2), (0,0,255), 2)
                for z in range(2):
                    if (len(detection_queue) > 0):
                        next_dot = detection_queue.pop(0)
                        img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)),  int(next_dot.diam/2), (0,0,255), 2)
                result[2] = result[2] + 1

        # [4]: nearest dot in 2*detection circle (+/- epsilon) / sqrt(2)
        elif (len(dots) > 3 and len(detection_queue) > 2 and 2*(detection_circle-eps)/np.sqrt(2) < dots[1].dist < 2*(detection_circle+eps)/np.sqrt(2)):
            detection_queue = calc_distance(current_dot, detection_queue)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(2*(detection_circle+eps)/np.sqrt(2)), (255,255,255), 1)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(current_dot.diam/2), (255,255,255), 2)
            for z in range(3):
                next_dot = detection_queue.pop(0)
                img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)),  int(next_dot.diam/2), (255,255,255), 2)
            result[3] = result[3] + 1

        # [6]: nearest dot in detection circle (+/- epsilon) / sqrt(2)
        elif (len(dots) > 5 and len(detection_queue) > 4 and (detection_circle-eps)/np.sqrt(2) < dots[1].dist < (detection_circle+eps)/np.sqrt(2)):
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int((detection_circle+eps)/np.sqrt(2)), (255,0,255), 1)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(current_dot.diam/2), (255,0,255), 2)
            detection_queue = calc_distance(current_dot, detection_queue)
            for z in range(5):
                next_dot = detection_queue.pop(0)
                img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)),  int(next_dot.diam/2), (255,0,255), 2)
            result[5] = result[5] + 1

        # when dot was not classified (side dots of 3 and 5)
        else:
            detection_queue.insert(0, current_dot)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(detection_circle+eps), (0,0,0), 1)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)),  int(detection_circle-eps), (0,0,0), 1)
        i = i + 1

    return img, result


def print_result(result):
    print("[1]:", result[0])
    print("[2]:", result[1])
    print("[3]:", result[2])
    print("[4]:", result[3])
    print("[5]:", result[4])
    print("[6]:", result[5])


def draw_result(img, result):
    cv2.putText(img, f"[1]: {result[0]}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[2]: {result[1]}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[3]: {result[2]}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[4]: {result[3]}", (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[5]: {result[4]}", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[6]: {result[5]}", (20, 180), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return img

def main():
    assets = get_assets()
    for file in assets:
        img_color, img_gray = load_picture(file)

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
        # detect dark dots on a white dice
        dark_dots_detector = cv2.SimpleBlobDetector_create(params)
        keypoints = dark_dots_detector.detect(img_gray)
        # detect white dots on a dark dice
        params.blobColor = 255
        white_dots_detector = cv2.SimpleBlobDetector_create(params)
        keypoints_white = white_dots_detector.detect(img_gray)

        img, result = detect_dices(img_color, keypoints)
        img = draw_result(img, result)
        print("----------------------------")
        print("file: ", file)
        print("k: ", len(keypoints))
        print_result(result)
        show_picture(img, 'Dice recognition')

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
