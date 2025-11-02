import os
import random
import time
from functools import partial  # noqa: F401

import cv2
import numpy as np
from queue import Empty

from pyscreeze import pixel

from final import actions  # noqa: F401
from final.utils import _enqueue_latest_move

_last_yellow_pos = None
_yellow_pos_confidence = 0


def reset_yellow_dot_tracking():
    global _last_yellow_pos, _yellow_pos_confidence
    _last_yellow_pos = None
    _yellow_pos_confidence = 0


def find_yellow_dot(minimap_img, hsv_range=None, area_thresh=8, max_area_thresh=200,
                    use_continuity=True, max_jump_distance=50, debug=False):
    """Find yellow dot on minimap. Returns (pos or None, red_flag).
    This is a cleaned-up version of the original implementation.
    """
    global _last_yellow_pos, _yellow_pos_confidence

    if minimap_img is None or minimap_img.size == 0:
        return None, False

    if hsv_range is None:
        lower_yellow = np.array([18, 100, 120], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
    else:
        lower_yellow = np.array(hsv_range[0], dtype=np.uint8)
        upper_yellow = np.array(hsv_range[1], dtype=np.uint8)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        try:
            cv2.imwrite("debug_minimap_yellow.png", minimap_img)
            cv2.imwrite("debug_minimap_yellow_mask.png", mask)
        except Exception:
            pass

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_valid = []
    for c in contours_red:
        area = cv2.contourArea(c)
        if area < area_thresh:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue
        red_valid.append(c)

    red_flag = len(red_valid) > 0

    if not contours:
        return (_last_yellow_pos if use_continuity and _yellow_pos_confidence > 3 else None), red_flag

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_thresh or area > max_area_thresh:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidates.append({
            'pos': (cx, cy),
            'area': area,
            'circularity': circularity,
            'contour': contour
        })

    if not candidates:
        return (_last_yellow_pos if use_continuity and _yellow_pos_confidence > 3 else None), False

    if len(candidates) == 1:
        best_candidate = candidates[0]
    else:
        if use_continuity and _last_yellow_pos is not None:
            for candidate in candidates:
                cx, cy = candidate['pos']
                dist = np.sqrt((cx - _last_yellow_pos[0])**2 + (cy - _last_yellow_pos[1])**2)
                candidate['distance'] = dist
            nearby_candidates = [c for c in candidates if c.get('distance', 9999) < max_jump_distance]
            if nearby_candidates:
                best_candidate = max(nearby_candidates, key=lambda c: c['circularity'])
            else:
                best_candidate = max(candidates, key=lambda c: c['circularity'])
                _yellow_pos_confidence = 0
        else:
            for candidate in candidates:
                area_score = 1.0 - abs(candidate['area'] - 30) / 100
                area_score = max(0, min(1, area_score))
                candidate['score'] = candidate['circularity'] * 0.7 + area_score * 0.3
            best_candidate = max(candidates, key=lambda c: c['score'])

    result_pos = best_candidate['pos']

    if _last_yellow_pos is not None:
        dist = np.sqrt((result_pos[0] - _last_yellow_pos[0])**2 + (result_pos[1] - _last_yellow_pos[1])**2)
        if dist < max_jump_distance:
            _yellow_pos_confidence = min(_yellow_pos_confidence + 1, 10)
        else:
            _yellow_pos_confidence = max(_yellow_pos_confidence - 2, 0)

    _last_yellow_pos = result_pos

    if debug:
        debug_img = minimap_img.copy()
        for i, candidate in enumerate(candidates):
            color = (0, 255, 0) if candidate == best_candidate else (0, 0, 255)
            cv2.circle(debug_img, candidate['pos'], 3, color, -1)
            cv2.putText(debug_img, f"{i}:{candidate['circularity']:.2f}",
                       (candidate['pos'][0] + 5, candidate['pos'][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        try:
            cv2.imwrite("debug_minimap_candidates.png", debug_img)
        except Exception:
            pass

    return result_pos, red_flag


def get_minimap_loc_size(img_frame):
    white = np.array([255, 255, 255])
    mask_white = cv2.inRange(img_frame, white, white)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_white, connectivity=8)
    for i in range(1, num_labels):
        x0, y0, rw, rh, area = stats[i]
        if rw < 100 or rh < 100:
            continue
        x1 = x0 + rw - 1
        y1 = y0 + rh - 1
        if not (np.all(img_frame[y0, x0:x0+rw] == white) and np.all(img_frame[y1, x0:x0+rw] == white)):
            continue
        if not (np.all(img_frame[y0:y0+rh, x0] == white) and np.all(img_frame[y0:y0+rh, x1] == white)):
            continue
        mask_minimap = np.any(img_frame[y0:y0+rh, x0:x0+rw] != white, axis=2).astype(np.uint8)
        coords = cv2.findNonZero(mask_minimap)
        if coords is None:
            continue
        x_minimap, y_minimap, w_minimap, h_minimap = cv2.boundingRect(coords)
        x_minimap += x0
        y_minimap += y0
        return x_minimap, y_minimap, w_minimap, h_minimap
    return None


def get_nearest_color_code(mini_img, mini_pos, move_queue=None, current_hwnd=None, debug=False, attck_queue=None, misc_queue=None, route_idx=0):
    """Scan around `mini_pos` and return (pixel_color, move_flag).

    If `move_queue` and `current_hwnd` are provided, this function will enqueue
    movement callables (partials bound with hwnd) directly into the queue using
    `_enqueue_latest_move` so callers (like `demo.main`) don't need to manage enqueueing.

    Parameters:
      - mini_img: BGR minimap crop
      - mini_pos: (x,y) estimated pos inside minimap
      - move_queue: optional Queue to receive movement partials
      - current_hwnd: optional hwnd bound into partials
      - debug: if True print diagnostic messages

    Returns:
      (pixel_color_or_None, move_flag)
      move_flag == True for rope (jump/climb), False otherwise.
    """
    x0, y0 = mini_pos
    h, w = mini_img.shape[:2]
    sr = 5
    x_min = max(0, x0 - sr)
    x_max = min(w, x0 + sr)
    y_min = max(0, y0 - sr)
    y_max = min(h, y0 + sr)


    pixels = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            pixel = mini_img[y, x]
            if np.array_equal(pixel, [0, 0, 255]):
                # rope indicator
                if debug:
                    print("✓ on route - 爬！")
                # enqueue move_target if we have queue and hwnd
                if move_queue is not None and current_hwnd is not None:
                    offset = random.randint(-2, 2)
                    p = partial(actions.move_target, current_hwnd, cur=mini_pos, target=(x + offset, y), is_rope=True, jump_dump=0.2)
                    #向上优于攻击
                    _enqueue_latest_move(move_queue, p)
                pixels.append(pixel)
            elif np.array_equal(pixel, [0, 255, 0]):
                if debug:
                    print("← move left - 向左移动！")
                if move_queue is not None and current_hwnd is not None:
                    p = partial(actions.move_left, current_hwnd, duration=0.5)
                    _enqueue_latest_move(move_queue, p)
            elif np.array_equal(pixel, [255, 0, 0]):
                if debug:
                    print("→ move right - 向右移动！")
                if move_queue is not None and current_hwnd is not None:
                    p = partial(actions.move_right, current_hwnd, duration=0.5)
                    _enqueue_latest_move(move_queue, p)
                pixels.append(pixel)
            elif np.array_equal(pixel, [255, 255, 0]):
                if debug:
                    print("→ jump right - 向右跳！")
                if move_queue is not None and current_hwnd is not None:
                    p = partial(actions.move_target, current_hwnd, cur=mini_pos, target=(x, y), is_rope=True, jump_dump=0.2, is_right=True)
                    _enqueue_latest_move(move_queue, p)
                pixels.append(pixel)
            elif np.array_equal(pixel, [255, 0, 255]):
                if debug:
                    print("→ jump left - 向左跳！")
                if move_queue is not None and current_hwnd is not None:
                    p = partial(actions.move_target, current_hwnd, cur=mini_pos, target=(x, y), is_rope=True, jump_dump=0.2, is_left=True)
                    _enqueue_latest_move(move_queue, p)
                pixels.append(pixel)
            elif np.array_equal(pixel, [0, 255, 255]):
                if debug:
                    print("到终点- 换路线")
                route_idx += 1

    return pixels, False, route_idx


def detect_minimap_dynamic(frame):
    """
    自动检测 MapleStory 小地图位置和大小（适应不同窗口分辨率）
    """
    h, w = frame.shape[:2]
    roi = frame[0:int(h * 0.55), 0:int(w * 0.55)]  # 左上角区域

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if area < 2000 or rw < 80 or rh < 80:
            continue
        if rw > w * 0.6 or rh > h * 0.6:
            continue

        ratio = rw / rh
        if 0.7 < ratio < 1.4:  # 大致接近方形
            roi_crop = gray[y:y+rh, x:x+rw]
            contrast = roi_crop.std()
            brightness = roi_crop.mean()
            edges_density = np.sum(edges[y:y+rh, x:x+rw] > 0) / area
            score = contrast * 0.6 + edges_density * 200 + brightness * 0.02
            candidates.append((score, x, y, rw, rh))

    if not candidates:
        return None

    # 选择得分最高的候选
    best = max(candidates, key=lambda c: c[0])
    _, x, y, rw, rh = best

    # 坐标偏移回原图
    return x, y, rw, rh

def detect_minimap_dynamic_v2(frame, debug=True):
    """
    自动检测 MapleStory 小地图位���和大小（适应不同窗口��辨率）
    """
    h, w = frame.shape[:2]
    roi = frame[0:int(h * 0.55), 0:int(w * 0.55)]  # 左上角区域（UI固定区）

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if area < 2000 or rw < 80 or rh < 80:
            continue
        if rw > w * 0.6 or rh > h * 0.6:
            continue

        ratio = rw / rh
        if 0.7 < ratio < 1.4:  # 小地图基本是方形
            roi_crop = gray[y:y+rh, x:x+rw]
            contrast = roi_crop.std()
            brightness = roi_crop.mean()
            edges_density = np.sum(edges[y:y+rh, x:x+rw] > 0) / area

            # 综合评分（可调系数）
            score = contrast * 0.6 + edges_density * 200 + brightness * 0.02
            candidates.append((score, x, y, rw, rh))

    if not candidates:
        print("⚠ 未检测到小地图")
        return None

    # 选择分数最高的候选
    best = max(candidates, key=lambda c: c[0])
    _, x, y, rw, rh = best

    # 坐标偏移回原图
    x_abs, y_abs = x, y

    return (x_abs, y_abs, rw, rh)


def prepare_route_map(dir):
    routes = []
    if os.path.exists(dir):
        routes = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().startswith(('route'))]

    return routes
