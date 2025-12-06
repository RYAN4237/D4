import time

import cv2
import numpy as np
import threading

from actions import *
from capture import *
from poe2 import a_star, actions
from poe2.a_star import mini_map_matching

from poe2.smart_stitch import SmartMinimapStitcher
from poe2.find_char import detect_and_annotate


def main():
    stitcher = SmartMinimapStitcher(
        x1=1081, y1=33,
        x2=1261, y2=176,
        name="Path of Exile 2"
    )
    stitch = threading.Thread(target=stitcher.run, daemon=True)
    stitch.start()

    capturer = CaptureScreen()
    capturer.get_hwnd("Path of Exile 2")


    # def on_mouse(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print(param[y, x])
    #
    # cv2.namedWindow("win")
    # img = cv2.imread("current_map_hsv.png")
    # img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    # cv2.setMouseCallback("win", on_mouse, img)
    # cv2.imshow("win", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    driver = Driver()
    driver.d_set_ini(r"C:\Repo\D4\poe2\driver.dll")

    while True:
        big_map = cv2.imread("final_smart_stitch_cropped.png")
        if big_map is None:
            print("No big map")
            time.sleep(1)
            continue
        big_map_copy = big_map.copy()
        current_map = capturer.capture(1081, 33, 1261, 176, 1, 1)
        current_map_hsv = cv2.cvtColor(current_map, cv2.COLOR_BGR2HSV)

        cv2.imshow("win", current_map_hsv)

        current_map_copy = current_map.copy()
        cx, cy = detect_and_annotate(current_map, [(25.0, 85.0, 69.0)])
        res = mini_map_matching(current_map, big_map, (cx, cy), 0.1, debug=False)
        big_map_three = a_star.merge_blue_into_binary(big_map_copy)
        path = a_star.a_star(start = res, goal = None, route_img=big_map_three, mini_img=current_map_copy)
        print(f"path start: {path[0] if path else 'empty'}")
        print(f"res (goal): {res}")
        driver.move(res[0], res[1], path, offset=0)
        cv2.circle(current_map_copy, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(big_map_copy, (res[0], res[1]), 10, (255, 0, 0), cv2.FILLED)
        cv2.imshow("Path of Exile 2", current_map_copy)
        cv2.imshow("Big Map", big_map_copy)
        cv2.imshow("big_map_three", big_map_three)
        if cv2.waitKey(100) == ord('q'):
            cv2.imwrite("current_map_hsv.png", current_map_hsv)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    time.sleep(2)
    main()