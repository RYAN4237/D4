import ctypes
import math
import os
import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


map_routes = {
            'name': 'Victoria Road - Henesys Hunting Ground II',
            'levels': {
                0: {'y_range': 130, 'platforms': [30, 120]},  # 底层
                1: {'y_range': 108, 'platforms': [30, 120]},  # 主层 (你当前位置)
                2: {'y_range': 85, 'platforms': [5, 120]}
            },
            'ropes': [
                {'x': 63, 'y_start': 130, 'y_end': 108},  # 主绳索
                {'x': 97, 'y_start': 108, 'y_end': 85},  # 辅助绳索
            ]
        }


def template_match_monster(img, sift, templates_cache):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_kp, img_des = sift.detectAndCompute(img_grey, None)
    bf =cv2.BFMatcher()
    good = []
    for file_obj in templates_cache:
        tem_kp, tem_des = sift.detectAndCompute(file_obj["gray"], None)
        if tem_des is None or img_des is None:
            continue
        matches = bf.knnMatch(img_des, tem_des, k=2)
        for m, n in matches:
            if m.distance < 0.4 * n.distance:
                good.append(img_kp[m.queryIdx].pt)

    return good




def template_match_character(img):
    """模板匹配检测人物"""
    template = cv2.imread(r"C:\Repo\D4\name.png")
    character_template = {
        'template': template,
        'gray': cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    }


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = character_template['gray']
    best_match = None
    best_confidence = 0

    # 多尺度匹配
    for scale in [1.0, 1.1, 1.2]:
        h, w = template_gray.shape
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        if scaled_w <= 0 or scaled_h <= 0:
            continue
        if scaled_h > img_gray.shape[0] or scaled_w > img_gray.shape[1]:
            continue

        scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
        result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_confidence:
            best_confidence = max_val
            center_x = max_loc[0] + scaled_w // 2
            center_y = max_loc[1] + scaled_h // 2
            character_y = center_y + scaled_h + 20

            best_match = {
                'position': max_loc,
                'center': (center_x, center_y),
                'character_pos': (center_x, character_y),
                'confidence': max_val,
                'scale': scale
            }

    if best_confidence > 0.65:
        # print(f"人物 {best_match}")
        h, w, _ = template.shape
        cv2.rectangle(img, (best_match['position'][0], best_match['position'][1]), (best_match['position'][0] + w, best_match['position'][1] + h), (0, 255, 255), 3)
    return best_match if best_confidence > 0.7 else {"center": (0, 0)}


def capture_window(window_title="MapleStory Worlds-Old School Maple"):
    """截取游戏窗口"""
    hwnd = win32gui.FindWindow(None, window_title)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
        (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img_bgr, hwnd

def calibrate_minimap_region(window_title="MapleStory Worlds-Old School Maple"):
    """校准小地图区域"""
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f"未找到窗口: {window_title}")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    # 小地图区域
    minimap_left = 72
    minimap_top = 140
    minimap_right = 242
    minimap_bottom = 300

    region = (minimap_left, minimap_top, minimap_right, minimap_bottom)
    return region

def find_yellow_dot(minimap_img):
    """在小地图中找到黄色点"""
    hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def press_key(hwnd, key_code, duration=0.05):
    """发送按键"""
    try:
        # print(f"发送按键 {key_code}")
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.02)

        win32api.keybd_event(key_code, 0, 0, 0)
        time.sleep(duration)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    except Exception as e:
        print(f"按键发送失败: {e}")

def attack_action(hwnd):
    """攻击动作"""
    # press_key(hwnd, win32con.VK_CONTROL)
    press_key(hwnd, win32con.VK_SPACE)

def move_left(hwnd, duration=0.5):
    press_key(hwnd, win32con.VK_LEFT, duration)

def move_right(hwnd,duration=0.5):
    press_key(hwnd,win32con.VK_RIGHT, duration)

def jump(hwnd):
    press_key(hwnd,win32con.VK_MENU, 0.05)  # Alt 是跳跃

def climb_up(hwnd, duration=2):
    press_key(hwnd, win32con.VK_UP, duration)

def move_target(hwnd, cur, target, duration=0.2):
    if cur[0] < target[0] - 5:
        move_right(hwnd, duration)
    elif cur[0] > target[0] + 5:
        move_left(hwnd, duration)
    else:
        jump(hwnd)
        climb_up(hwnd)
        return True

def auto_route(sift, templates_cache):
    # print(f"开始自动寻路: {map_data['name']}")
    img, hwnd = capture_window()
    levels = map_routes['levels']
    ropes = map_routes['ropes']

    route = [30, 90]
    level_pos = 0
    region = calibrate_minimap_region()
    minimap_img = img[region[1]:region[3], region[0]:region[2]]
    min_pos = find_yellow_dot(minimap_img)
    # 创建可调整大小的窗口（必须在 imshow 之前）
    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    # 将窗口尺寸设为 800x600（像素）
    cv2.resizeWindow("win", 800, 600)
    cv2.imshow("win", img)



    while True:
        for level in levels:
            if min_pos[1] == levels[level]['y_range']:
                route = levels[level]['platforms']
                level_pos = level

        print(f"route {route}, level_pos {level_pos}")

        while True:
            img, hwnd = capture_window()
            big_pos = template_match_character(img)["center"]
            goods = template_match_monster(img, sift, templates_cache)
            minimap_img = img[region[1]:region[3], region[0]:region[2]]
            min_pos = find_yellow_dot(minimap_img)
            cv2.circle(img, big_pos, 10, (255, 0, 0), 2)
            cv2.circle(img, min_pos, 10, (255, 0, 0), 2)
            print(f"小地圖 {min_pos}, 大地圖 {big_pos}")


            goods = np.array(goods)
            cx, cy = big_pos
            indices = np.where(
                (goods[:, 0] > cx - 150) & (goods[:, 0] < cx + 150) &
                (goods[:, 1] > cy - 100) & (goods[:, 1] < cy + 100)
            )
            filtered_goods = goods[indices]
            for good in filtered_goods:
                cv2.circle(img, (int(good[0]), int(good[1])), 5, (255, 0, 0), -1)
                attack_action(hwnd)

            move_right(hwnd)
            # 更新显示并处理按键（在每帧都处理）
            cv2.imshow("win", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if min_pos[0] <= route[1] - 10:
                break


        while True:
            img, hwnd = capture_window()
            big_pos = template_match_character(img)["center"]
            goods = template_match_monster(img, sift, templates_cache)
            minimap_img = img[region[1]:region[3], region[0]:region[2]]
            min_pos = find_yellow_dot(minimap_img)
            cv2.circle(img, big_pos, 10, (255, 0, 0), 2)
            cv2.circle(img, min_pos, 10, (255, 0, 0), 2)
            print(f"小地圖 {min_pos}, 大地圖 {big_pos}")

            goods = np.array(goods)
            cx, cy = big_pos
            indices = np.where(
                (goods[:, 0] > cx - 150) & (goods[:, 0] < cx + 150) &
                (goods[:, 1] > cy - 100) & (goods[:, 1] < cy + 100)
            )
            filtered_goods = goods[indices]
            for good in filtered_goods:
                cv2.circle(img, (int(good[0]), int(good[1])), 5, (255, 0, 0), -1)
                attack_action(hwnd)

            move_left(hwnd)
            # 更新显示并处理按键（在每帧都处理）
            cv2.imshow("win", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if min_pos[0] <= route[0] + 10:
                break

        if level_pos in [0, 1]:
            rope_x = ropes[level_pos]['x']
            while True:
                img, hwnd = capture_window()
                minimap_img = img[region[1]:region[3], region[0]:region[2]]
                min_pos = find_yellow_dot(minimap_img)
                if move_target(hwnd, min_pos, (rope_x, 0)):
                    break

        time.sleep(1)
        # 主循环退出检测（ESC handled above). 让主循环短暂休息后继续下一轮
        if cv2.waitKey(1) & 0xFF == 27:
            break


    # 循环外一次性销毁窗口
    cv2.destroyAllWindows()

def load_templates_cache(templates_dir, sift):
    cache = []
    files = sorted([f for f in os.listdir(templates_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    for fn in files:
        path = os.path.join(templates_dir, fn)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            des = des.astype(np.float32)
        cache.append({'name': fn, 'img': img, 'gray': gray, 'kp': kp, 'des': des, 'shape': img.shape[:2]})
    return cache

if __name__ == '__main__':
    time.sleep(3)
    sift = cv2.SIFT.create()
    templates_cache = load_templates_cache(r"C:\Repo\D4\monsters", sift)
    auto_route(sift, templates_cache)







