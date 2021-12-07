import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

RESIZED_WIDTH = 700
DC_FACTOR = 1.8
EPS_FACTOR = 0.25
PIC_IN_ROW = 5


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
    resize_ratio = RESIZED_WIDTH / width
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
        dot.dist = np.sqrt((current_dot.x - dot.x) ** 2 + (current_dot.y - dot.y) ** 2)
    dots.sort(key=lambda k: k.dist)
    return dots


def detect_dices(img, key_points):
    result = [0] * 6
    i = 0
    dots = []  # all dots detected
    detection_queue = []  # dots for calculations
    temp = []

    if len(key_points) == 0:
        return img, result

    for k in key_points:
        img = cv2.circle(img, (int(k.pt[0]), int(k.pt[1])), int(k.size / 2), (0, 255, 0), 2)
        dots.append(Dot(k.pt[0], k.pt[1], k.size))
        temp.append(k.size)
    detection_queue = dots[:]

    # percentiles used to threshold dot size and minimize their impact for detection circle
    q1 = np.percentile(temp, 30)    # lower limit
    q3 = np.percentile(temp, 90)    # upper limit

    while len(detection_queue) > 0 and len(dots) > 0 and i < 1000:
        current_dot = detection_queue.pop()

        if current_dot.diam < q1:
            detection_circle = DC_FACTOR * q1
        elif current_dot.diam > q3:
            detection_circle = DC_FACTOR * q3
        else:
            detection_circle = DC_FACTOR * current_dot.diam
        eps = EPS_FACTOR * current_dot.diam
        dots = calc_distance(current_dot, dots)  # dots[0] = current_dot
        enclosing_points = [(current_dot.x, current_dot.y)]

        # [1]: nearest dot > 2*detection circle
        if len(dots) == 1 or dots[1].dist > 2 * (detection_circle + eps * 1.3):
            result[0] = result[0] + 1
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(2 * (detection_circle - eps)), (255, 0, 0), 2)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam / 2), (255, 0, 0), 2)

        # [2]: nearest dot in 2*detection_circle (+/- epsilon)
        elif len(dots) > 1 and 2 * (detection_circle - eps) < dots[1].dist < 2 * (detection_circle + eps * 1.3):
            detection_queue = calc_distance(current_dot, detection_queue)
            next_dot = detection_queue.pop(0)
            enclosing_points.append((next_dot.x, next_dot.y))
            enclosing_points = np.array(enclosing_points, dtype=np.int32)
            result[1] = result[1] + 1

            (x, y), r = cv2.minEnclosingCircle(enclosing_points)
            img = cv2.circle(img, (int(x), int(y)), int(r + current_dot.diam), (255, 255, 0), 2)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam / 2), (255, 255, 0), 2)
            img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam / 2), (255, 255, 0), 2)

        # [3] or [5]: two nearest dots in detection circle  
        elif len(dots) > 2 and len(detection_queue) > 1 and (detection_circle - eps) < dots[1].dist < (detection_circle + eps) and (detection_circle - eps) < dots[2].dist < (detection_circle + eps):
            # [5]: third nearest dot is also in detection circle (+/- epsilon)
            if len(dots) > 4 and len(detection_queue) > 3 and (detection_circle - eps) < dots[3].dist < (detection_circle + eps):
                detection_queue = calc_distance(current_dot, detection_queue)

                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam/2), (0, 255, 255), 2)
                for z in range(4):
                    if len(detection_queue) > 0:
                        next_dot = detection_queue.pop(0)
                        enclosing_points.append((next_dot.x, next_dot.y))
                        img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam/2), (0, 255, 255), 2)
                enclosing_points = np.array(enclosing_points, dtype=np.int32)
                (x, y), r = cv2.minEnclosingCircle(enclosing_points)
                img = cv2.circle(img, (int(x), int(y)), int(r + current_dot.diam), (0, 255, 255), 2)

                result[4] = result[4] + 1
            # [3]: third nearest dot outside detection circle
            elif dots[3].dist > (detection_circle + eps):
                detection_queue = calc_distance(current_dot, detection_queue)
                img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam/2), (0, 0, 255), 2)
                for z in range(2):
                    if len(detection_queue) > 0:
                        next_dot = detection_queue.pop(0)
                        enclosing_points.append((next_dot.x, next_dot.y))
                        img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam/2), (0, 0, 255), 2)

                enclosing_points = np.array(enclosing_points, dtype=np.int32)
                (x, y), r = cv2.minEnclosingCircle(enclosing_points)
                img = cv2.circle(img, (int(x), int(y)), int(r + current_dot.diam), (0, 0, 255), 2)

                result[2] = result[2] + 1

        # [4]: nearest dot in 2*detection circle (+/- epsilon) / sqrt(2)
        elif len(dots) > 3 and len(detection_queue) > 2 and 2 * (detection_circle - eps) / np.sqrt(2) < dots[1].dist < 2 * (detection_circle + eps) / np.sqrt(2):
            detection_queue = calc_distance(current_dot, detection_queue)
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam/2), (255, 255, 255), 2)
            for z in range(3):
                next_dot = detection_queue.pop(0)
                enclosing_points.append((next_dot.x, next_dot.y))
                img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam / 2), (255, 255, 255), 2)

            enclosing_points = np.array(enclosing_points, dtype=np.int32)
            (x, y), r = cv2.minEnclosingCircle(enclosing_points)
            img = cv2.circle(img, (int(x), int(y)), int(r + current_dot.diam), (255, 255, 255), 2)

            result[3] = result[3] + 1

        # [6]: nearest dot in detection circle (+/- epsilon) / sqrt(2)
        elif len(dots) > 5 and len(detection_queue) > 4 and (detection_circle - eps) / np.sqrt(2) < dots[1].dist < (detection_circle + eps) / np.sqrt(2):
            img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(current_dot.diam / 2), (255, 0, 255), 2)
            detection_queue = calc_distance(current_dot, detection_queue)
            for z in range(5):
                next_dot = detection_queue.pop(0)
                enclosing_points.append((next_dot.x, next_dot.y))
                img = cv2.circle(img, (int(next_dot.x), int(next_dot.y)), int(next_dot.diam / 2), (255, 0, 255), 2)

            enclosing_points = np.array(enclosing_points, dtype=np.int32)
            (x, y), r = cv2.minEnclosingCircle(enclosing_points)
            img = cv2.circle(img, (int(x), int(y)), int(r + current_dot.diam), (255, 0, 255), 2)

            result[5] = result[5] + 1

        # when dot was not classified (side dots of 3 and 5)
        else:
            detection_queue.insert(0, current_dot)
            # img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(detection_circle + eps), (0, 0, 0), 1)
            # img = cv2.circle(img, (int(current_dot.x), int(current_dot.y)), int(detection_circle - eps), (0, 0, 0), 1)
        i = i + 1

    return img, result


def print_result(result):
    print("[1]:", result[0])
    print("[2]:", result[1])
    print("[3]:", result[2])
    print("[4]:", result[3])
    print("[5]:", result[4])
    print("[6]:", result[5])


def draw_result(img, result, k):
    cv2.putText(img, f"[1]: {result[0]}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[2]: {result[1]}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[3]: {result[2]}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[4]: {result[3]}", (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[5]: {result[4]}", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"[6]: {result[5]}", (20, 180), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, f" k: {k}", (600, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    k_sum = 0
    for i in range(6):
        k_sum += result[i] * (i + 1)
    cv2.putText(img, f"ks: {k_sum}", (600, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def main():
    assets = get_assets()
    plt.figure(figsize=(24, 12), dpi=300)

    for i, file in enumerate(assets):
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
        # image thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        # detect dark dots on a white dice
        dark_dots_detector = cv2.SimpleBlobDetector_create(params)
        key_points = dark_dots_detector.detect(img_gray)
        # detect white dots on a dark dice
        params.blobColor = 255
        white_dots_detector = cv2.SimpleBlobDetector_create(params)
        key_points_white = white_dots_detector.detect(img_gray)

        img, result = detect_dices(img_color, key_points)
        img = draw_result(img, result, len(key_points))
        print("----------------------------")
        print("file: ", file)
        print("k: ", len(key_points))
        print_result(result)

        ax = plt.subplot2grid((len(assets) // PIC_IN_ROW + 1, PIC_IN_ROW), (i // PIC_IN_ROW, i % PIC_IN_ROW))
        ax.axis('off')
        ax.imshow((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        show_picture(img, 'Dice recognition')

    plt.savefig('dice_result.pdf')
    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
