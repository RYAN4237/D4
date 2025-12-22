import time

import cv2
import numpy as np

from actions import *
from capture import *
from poe2 import a_star, actions
from poe2.a_star import mini_map_matching
from poe2.attack import attack_nearby_enemies
from poe2.draft_idea.path_test import PathTest
from poe2.map_utils import MapUtils

from poe2.stitch_new import SmartMinimapStitcher
from poe2 import stitch_new
from poe2.find_char import detect_and_annotate
import multiprocessing as mp
import keyboard

from poe2.visited_recorder import VisitedRecorder

stop_flag = False

def on_f1_press():
    """F1按键回调函数"""
    global stop_flag
    print("\n[提示] 检测到F1按键，正在停止程序...")
    stop_flag = True

def worker_process(task_queue, result_queue):
    recorder = VisitedRecorder()
    while True:
        start_time = int(time.time() * 1000)
        task = task_queue.get()    # 阻塞等待任务
        if task == "STOP":
            break

        res, weight_grid, current_map_copy = task

        # 跑 A*
        path = a_star.a_star_new(
            start=res,
            goal=None,
            route_img=weight_grid,        # 注意: 传 float32 grid，不要传 uint8 图
            mini_img=current_map_copy,
            recorder=recorder
        )
        # print(f"worker process a_star time = {int(time.time() * 1000) - start_time} ms")
        result_queue.put(path)



def main():
    start_t = int(time.time() * 1000)
    keyboard.add_hotkey('f1', on_f1_press)
    task_queue = mp.Queue(maxsize=1)  # 永远保持最新任务
    result_queue = mp.Queue(maxsize=1)

    # 创建拼接结果队列
    stitch_queue = mp.Queue(maxsize=1)

    # stitcher = SmartMinimapStitcher(
    #     x1=1081, y1=33,
    #     x2=1261, y2=185,
    #     name="Path of Exile 2"
    # )

    driver = Driver()
    driver.d_set_ini(r"C:\Repo\D4\poe2\driver.dll")

    # 启动拼接进程
    stitch_process = mp.Process(
        target=stitch_new.stitch_worker_process,
        args=(stitch_queue, 1081, 33, 1261, 185, "Path of Exile 2"),
        daemon=True
    )
    stitch_process.start()

    # 启动 A* 工作进程
    # print(f"start worker process time = {int(time.time() * 1000) - start_t} ms")
    fight = mp.Process(target=worker_process, args=(task_queue, result_queue))
    fight.daemon = True
    fight.start()
    # print(f"worker process started time = {int(time.time() * 1000) - start_t} ms")

    capturer = CaptureScreen()
    capturer.get_hwnd("Path of Exile 2")
    path_test = PathTest()

    pre_res = None
    offset = 40
    big_map = None  # 缓存最新的 big_map

    while not stop_flag:
        # 从队列获取最新的拼接结果
        try:
            while not stitch_queue.empty():
                big_map = stitch_queue.get_nowait()
        except:
            pass

        if big_map is None:
            print("No big map")
            time.sleep(0.1)
            continue
        big_map_copy = big_map.copy()
        current_map = capturer.capture(1090, 35, 1260, 185, 1, 1)
        current_map_hsv = cv2.cvtColor(current_map, cv2.COLOR_BGR2HSV)
        current_map_copy = current_map.copy()
        cx, cy = detect_and_annotate(current_map, [(25.0, 85.0, 69.0)])
        res = mini_map_matching(current_map, big_map, (cx, cy), 0.1, debug=False)
        if pre_res is not None and abs(res[0] - pre_res[0]) > offset and abs(res[1] - pre_res[1]) > offset:
            res = pre_res
        big_map_three = MapUtils.merge_blue_into_binary(big_map_copy)
        # print(f"task queue put time = {int(time.time() * 1000) - start_t} ms")
        task_queue.put((res, big_map_three, current_map))
        print(f"Matched position: {res}")
        try:
            path = result_queue.get(timeout=0.1)
            print(f"Path length: {len(path)}")
            for i in range(len(path) - 1):
                cv2.line(big_map_three, path[i], path[i + 1], (127, 255, 127), 2)
            driver.move(res[0], res[1], path, offset=15)
        except:
            print("No path result yet")
        # print(f"main loop time = {int(time.time() * 1000) - start_t} ms")
        attack_nearby_enemies(driver, capturer)
        cv2.circle(current_map_copy, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(big_map_copy, (res[0], res[1]), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(big_map_three, (cx, cy), 5, (0, 0, 0), -1)
        cv2.imshow("Path of Exile 2", current_map_copy)
        cv2.imshow("Big Map", big_map_copy)
        cv2.imshow("big_map_three", big_map_three)
        # cv2.imshow("hsv map", current_map_hsv)
        pre_res = res
        time.sleep(0.05)
        if cv2.waitKey(100) == ord('q'):
            cv2.imwrite("current_map_hsv.png", current_map_hsv)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    time.sleep(2)
    main()