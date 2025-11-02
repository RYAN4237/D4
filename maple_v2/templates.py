import os
from functools import partial
import threading
import time
import pathlib

import cv2
import numpy as np

from final import actions
from final.utils import _enqueue_latest_move


def load_templates_cache(templates_dir, mask_color=(0, 255, 0), green_tol_fallback=True):
    cache = []
    files = sorted([f for f in os.listdir(templates_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    for fn in files:
        path = os.path.join(templates_dir, fn)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # try to compute mask by exact green or tolerant green
        mask = None
        # exact green
        mask_exact = np.all(img == np.array(mask_color, dtype=np.uint8), axis=2).astype(np.uint8) * 255
        if np.count_nonzero(mask_exact) > 0:
            mask = mask_exact
        else:
            b, g, r = cv2.split(img)
            mask_tol = ((g > 150) & (g > r + 30) & (g > b + 30)).astype(np.uint8) * 255
            if np.count_nonzero(mask_tol) > 0:
                mask = mask_tol
            else:
                diff = np.abs(img.astype(np.int16) - np.array(mask_color, dtype=np.int16))
                diff_sum = diff.sum(axis=2)
                mask_final = (diff_sum > 30).astype(np.uint8) * 255
                mask = mask_final

        mask = (mask > 0).astype(np.uint8) * 255

        cache.append({
            'name': fn,
            'img': img,
            'gray': gray,
            'mask': mask,
            'shape': img.shape[:2]
        })
    return cache


def template_match_character(img, template_path=None, min_confidence=0.7):
    if template_path is None:
        template_path = r"C:\Repo\D4\name.png"
    tpl = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if tpl is None:
        return {"center": (0, 0)}
    template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_confidence = 0
    for scale in [0.8, 1.0, 1.2]:
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
    return best_match if best_confidence > min_confidence else {"center": (0, 0)}


def template_match_monster(img, sift, templates_cache, ch_pos, is_left=False, match_ratio=0.5, matcher_norm=cv2.NORM_L2, key_queue=None, attack_queue=None, move_queue=None, current_hwnd=None, debug=False):
    # crop region near character and compute offsets
    try:
        if ch_pos is None:
            print("[template_match_monster] no ch_pos provided", flush=True)
            return []
        cx, cy = ch_pos
        y0 = max(0, int(cy) - 200)
        y1 = int(cy) + 30
        if is_left:
            x0 = max(0, int(cx) - 300)
            x1 = int(cx)
        else:
            x0 = int(cx)
            x1 = int(cx) + 300

        # clamp to image bounds
        h, w = img.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))

        if x1 <= x0 or y1 <= y0:
            print(f"[template_match_monster] invalid crop coords x0={x0} x1={x1} y0={y0} y1={y1}", flush=True)
            return []

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            print("[template_match_monster] empty crop", flush=True)
            return []

        img_grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # ensure correct dtype for OpenCV feature detectors
        if img_grey.dtype != np.uint8:
            try:
                img_grey = img_grey.astype(np.uint8)
            except Exception:
                pass
        try:
            kp_img, des_img = sift.detectAndCompute(img_grey, None)
        except cv2.error as e:
            print(f"[template_match_monster] OpenCV detectAndCompute error: {e}", flush=True)
            return []

        bf = cv2.BFMatcher(matcher_norm)
        goods = []
        templ_count = 0
        for file_obj in templates_cache:
            templ_count += 1
            tem_gray = file_obj['gray']
            for size in [0.8, 1.0, 1.2]:
                tem_gray_resized = cv2.resize(tem_gray, None, fx=size, fy=size)
                tem_gray = tem_gray_resized
                h, w = tem_gray.shape
                try:
                    kp_t, des_t = sift.detectAndCompute(tem_gray, None)
                except cv2.error as e:
                    print(f"[template_match_monster] OpenCV detectAndCompute on template failed: {e}", flush=True)
                    continue
                if des_t is None:
                    continue
                try:
                    matches = bf.knnMatch(des_img, des_t, k=2)
                except Exception as e:
                    print(f"[template_match_monster] knnMatch failed: {e}", flush=True)
                    continue
                for m_n in matches:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < match_ratio * n.distance:
                        pt = kp_img[m.queryIdx].pt
                        # convert to original image coordinates
                        abs_x = x0 + int(pt[0])
                        abs_y = y0 + int(pt[1])
                        # draw on original image
                        try:
                            cv2.rectangle(img, (abs_x, abs_y), (abs_x + w, abs_y + h), (255, 0, 0), 2)
                        except Exception:
                            pass
                        goods.append((abs_x, abs_y))

        # Debug: print detected goods count and thread info (safe in multithreaded context)
        # try:
        #     print(f"[template_match_monster] thread={threading.current_thread().name} goods={len(goods)} templates_tried={templ_count}", flush=True)
        # except Exception:
        #     pass

        # 攻击逻辑（如果提供了 key_queue，则入队）
        if (len(goods) > 0):
            try:
                # Prefer enqueuing movement into move_queue and attacks into attack_queue.
                # If current_hwnd is provided, bind it in the partials so worker can call without args.
                # Movement
                if current_hwnd is not None:
                    # 转身 攻击
                    if is_left:
                        p =partial(actions.move_left, current_hwnd, duration=0.05)
                        _enqueue_latest_move(attack_queue, p)
                    else:
                        p = partial(actions.move_right, current_hwnd, duration=0.05)
                        _enqueue_latest_move(attack_queue, p)
                    for good in goods:
                        p = partial(actions.attack_action, current_hwnd)
                        _enqueue_latest_move(attack_queue, p)

            except Exception:
                # ignore queue exceptions
                pass
        return goods
    except Exception as e:
        # top-level safeguard so background thread doesn't crash silently
        try:
            import traceback
            print(f"[template_match_monster] exception: {e}", flush=True)
            traceback.print_exc()
        except Exception:
            pass
        return []

def yolo_detect_monsters(img, model, min_confidence=0.6, attack_queue=None, current_hwnd=None):
    results = model.predict(img, conf=min_confidence, verbose=False)
    player = []
    monsters = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if cls_id == 2:
                player.append((center_x, center_y))
            elif cls_id == 1:
                monsters.append((center_x, center_y))
            # draw rectangle
            try:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception:
                pass

    # 攻击逻辑（如果提供了 attack_queue，则入队）
    if (len(monsters) > 0):
        try:
            if current_hwnd is not None:
                for monster in monsters:
                    px, py = player[0]
                    mx, my = monster
                    if px - 200 <= mx <= px + 200 and py - 150 <= my <= py + 50:
                        # 我用剑雨所以不需要转向
                        p = partial(actions.attack_action, current_hwnd)
                        _enqueue_latest_move(attack_queue, p)
        except Exception:
            # ignore queue exceptions
            pass
    return monsters